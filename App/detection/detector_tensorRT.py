import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit # PyCUDA 초기화 및 컨텍스트 관리를 위해 필요합니다.
                      # 하지만, 명시적 컨텍스트 관리를 위해 주석 처리하거나 제거할 수 있습니다.
                      # 현재 코드는 명시적 컨텍스트 관리를 하므로 이 줄은 필요 없습니다.

import torch # postprocess의 sigmoid 함수에 사용될 수 있습니다. (현재 np.exp 사용)
import os
import torch.nn.functional as F # 사용되지 않으므로 제거 가능

# TensorRT 로거 설정 (오류 메시지를 콘솔에 출력)
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

class Detector:
    def __init__(self, engine_path="resources/models/v2_rt_test1.engine", conf_threshold=0.5, iou_threshold=0.45):
        print(f"DEBUG: Initializing Detector with engine: {engine_path}")
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

        # ONNX 모델의 입력 크기에 맞춰 조정해야 합니다.
        # 실제 .engine 파일의 입력 크기를 확인하여 이 값을 설정해야 합니다.
        # 예를 들어, YOLOv8 모델의 입력 크기가 640x640 이면
        self.input_height = 192 # ONNX 모델의 입력 높이에 맞춰 설정
        self.input_width = 320  # ONNX 모델의 입력 너비에 맞춰 설정
        self.padding_color = 114

        # --- CUDA 컨텍스트 초기화 ---
        try:
            # pycuda.autoinit을 사용하지 않는 경우, 여기서 직접 컨텍스트를 생성하고 관리합니다.
            # 하지만 pycuda.autoinit을 import하면 기본적으로 컨텍스트가 생성됩니다.
            # 만약 autoinit을 쓰지 않고 직접 primary context를 쓰려면 아래처럼 합니다.
            # self.cuda_device = cuda.Device(0)
            # self.cuda_context = self.cuda_device.retain_primary_context()
            # self.cuda_context.push() # __init__ 내에서 CUDA 작업 시작
            # print("DEBUG: CUDA Primary Context pushed for initialization.")

            # 이미 pycuda.autoinit이 활성화되어 있다면, 별도 push/pop 없이 현재 컨텍스트를 사용합니다.
            # 하지만 명시적 관리를 선호한다면 autoinit을 지우고 위 코드를 사용해야 합니다.
            # 이 예제에서는 autoinit을 지운다고 가정하고 명시적 push/pop을 유지합니다.
            # 만약 pycuda.autoinit이 활성화된 상태라면 아래 push()는 추가적인 스택 푸시가 됩니다.
            # 현재 코드에서 pycuda.autoinit이 주석 처리되었으므로, 명시적으로 컨텍스트를 생성합니다.
            self.cuda_device = cuda.Device(0)
            self.cuda_context = self.cuda_device.retain_primary_context()
            self.cuda_context.push() # __init__ 초기화 작업 위해 컨텍스트 활성화
            print("DEBUG: CUDA Context pushed for initialization.")


        except Exception as e:
            error_msg = f"오류: CUDA 컨텍스트 초기화/활성화 중 예외 발생: {e}"
            print(error_msg)
            raise RuntimeError(error_msg)
        # --- CUDA 컨텍스트 초기화 끝 ---

        # --- TensorRT 엔진 로드 ---
        if not os.path.exists(engine_path):
            error_msg = f"오류: TensorRT 엔진 파일이 다음 경로에 없습니다: {engine_path}"
            print(error_msg)
            if hasattr(self, 'cuda_context') and self.cuda_context:
                self.cuda_context.pop() # 오류 시 컨텍스트 정리
            raise FileNotFoundError(error_msg)

        try:
            with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
                self.engine = runtime.deserialize_cuda_engine(f.read())
        except Exception as e:
            error_msg = f"오류: TensorRT 엔진 로드 중 예외 발생: {e}"
            print(error_msg)
            if hasattr(self, 'cuda_context') and self.cuda_context:
                self.cuda_context.pop() # 오류 시 컨텍스트 정리
            raise RuntimeError(error_msg)

        if not self.engine:
            error_msg = "오류: TensorRT 엔진 로드 실패."
            print(error_msg)
            if hasattr(self, 'cuda_context') and self.cuda_context:
                self.cuda_context.pop() # 오류 시 컨텍스트 정리
            raise RuntimeError(error_msg)

        self.context = self.engine.create_execution_context()
        if not self.context:
            error_msg = "오류: TensorRT 실행 컨텍스트 생성 실패."
            print(error_msg)
            if hasattr(self, 'cuda_context') and self.cuda_context:
                self.cuda_context.pop() # 오류 시 컨텍스트 정리
            raise RuntimeError(error_msg)

        # --- 입력/출력 바인딩 설정 ---
        self.input_name = None
        self.output_names = []
        self.bindings = [None] * self.engine.num_io_tensors
        self.input_binding_idx = -1

        self.output_host_mem = {}  # CPU 출력 버퍼
        self.output_device_mem = {} # GPU 출력 버퍼
        self.output_shapes = {} # 출력 텐서 형태

        for i in range(self.engine.num_io_tensors):
            binding_name = self.engine.get_tensor_name(i)
            actual_shape = self.context.get_tensor_shape(binding_name)
            binding_dtype = trt.nptype(self.engine.get_tensor_dtype(binding_name))
            
            if self.engine.get_tensor_mode(binding_name) == trt.TensorIOMode.INPUT:
                self.input_name = binding_name
                self.input_binding_idx = i
                
                # 입력 텐서 형태 확인 (BCHW)
                if len(actual_shape) == 4:
                    # self.input_batch_size = actual_shape[0] # ONNXRuntime 코드에서는 사용되지 않음
                    # self.input_channel = actual_shape[1]
                    self.input_height = actual_shape[2] # 엔진으로부터 입력 높이/너비 가져오기
                    self.input_width = actual_shape[3]
                else:
                    raise ValueError(f"예상치 못한 입력 텐서 형태: {actual_shape}, 이름: {binding_name}")
                
                # 입력 GPU 메모리 할당
                self.input_device_mem = cuda.mem_alloc(trt.volume(actual_shape) * np.dtype(binding_dtype).itemsize)
                self.bindings[i] = int(self.input_device_mem)
            else: # OUTPUT
                self.output_names.append(binding_name)
                self.output_shapes[binding_name] = actual_shape
                
                # 출력 CPU 및 GPU 메모리 할당
                host_mem = cuda.pagelocked_empty(trt.volume(actual_shape), dtype=binding_dtype)
                self.output_host_mem[binding_name] = host_mem
                device_mem = cuda.mem_alloc(host_mem.nbytes)
                self.output_device_mem[binding_name] = device_mem
                self.bindings[i] = int(device_mem)

        if self.input_name is None:
            if hasattr(self, 'cuda_context') and self.cuda_context:
                self.cuda_context.pop()
            raise RuntimeError("엔진에서 입력 바인딩을 찾을 수 없습니다.")
        if not self.output_names:
            if hasattr(self, 'cuda_context') and self.cuda_context:
                self.cuda_context.pop()
            raise RuntimeError("엔진에서 출력 바인딩을 찾을 수 없습니다.")
        
        print(f"TensorRT 모델 입력: 이름={self.input_name}, 형태=(H:{self.input_height}, W:{self.input_width})")
        for name in self.output_names:
            print(f"TensorRT 모델 출력: 이름={name}, 형태={self.output_shapes[name]}")

        self.class_names = [
            "forklift-right", "forklift-left", "forklift-vertical",
            "forklift-horizontal", "person", "object",
        ]
        self.num_classes = len(self.class_names)
        
        # __init__이 끝날 때 컨텍스트 비활성화
        if hasattr(self, 'cuda_context') and self.cuda_context:
            self.cuda_context.pop()
            print("DEBUG: CUDA Context popped after initialization.")

    @staticmethod
    def sigmoid(x):
        # ONNXRuntime 코드에서 np.exp를 사용하여 시그모이드 구현
        return 1 / (1 + np.exp(-x))

    def preprocess(self, frame):
        orig_h, orig_w = frame.shape[:2]

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 현재 input_width와 input_height를 사용합니다.
        # 비율 유지를 위해 scale 계산
        scale = min(self.input_width / orig_w, self.input_height / orig_h)
        
        target_width_resized = int(orig_w * scale)
        target_height_resized = int(orig_h * scale)

        resized_frame = cv2.resize(rgb_frame, (target_width_resized, target_height_resized), interpolation=cv2.INTER_LINEAR)
        
        # 패딩 계산 (모델 입력 크기에 맞춰)
        pad_top = (self.input_height - target_height_resized) // 2
        pad_bottom = self.input_height - target_height_resized - pad_top
        pad_left = (self.input_width - target_width_resized) // 2
        pad_right = self.input_width - target_width_resized - pad_left
        
        padded_img = cv2.copyMakeBorder(
            resized_frame,
            pad_top, pad_bottom, pad_left, pad_right,
            cv2.BORDER_CONSTANT,
            value=(self.padding_color, self.padding_color, self.padding_color)
        )
        
        img_input = padded_img.transpose(2, 0, 1) # HWC -> CHW
        img_input = np.expand_dims(img_input, axis=0).astype(np.float32) / 255.0
        img_input = np.ascontiguousarray(img_input) # 메모리 연속성 확보

        return img_input, scale, (pad_left, pad_top) # 패딩은 (left, top)으로 전달

    def detect_objects(self, frame):
        # detect_objects 호출 시 컨텍스트 다시 활성화
        if hasattr(self, 'cuda_context') and self.cuda_context:
            self.cuda_context.push()
            # print("DEBUG: CUDA Context pushed in detect_objects.") # 디버깅용

        try:
            img_input, scale_factor, pad_offsets = self.preprocess(frame)
            original_h, original_w = frame.shape[:2]

            # GPU로 입력 데이터 복사
            cuda.memcpy_htod(self.bindings[self.input_binding_idx], img_input)

            # 추론 실행
            self.context.execute_v2(bindings=self.bindings)

            # GPU에서 CPU로 출력 데이터 복사
            trt_engine_outputs = [] # TensorRT는 출력 순서를 인덱스로 다룸
            for name in self.output_names:
                cuda.memcpy_dtoh(self.output_host_mem[name], self.output_device_mem[name])
                # ONNXRuntime과 동일한 형태 (squeeze(0)되지 않은 배치 차원 포함)로 전달
                # TensorRT 출력은 이미 배치 차원이 포함되어 있을 수 있으므로 .reshape(output_shape)
                # 이 예제에서는 outputs[0].squeeze(0)을 가정하고 있으므로, 그대로 전달합니다.
                trt_engine_outputs.append(self.output_host_mem[name].reshape(self.output_shapes[name]))

            # outputs[0]이 main_output (boxes, scores, masks)이라고 가정
            # TensorRT 모델의 출력 구조에 따라 이 부분은 조정될 수 있습니다.
            # YOLOv8 ONNX 모델을 TensorRT로 변환했을 때 일반적으로 `output0` (detections)와 `output1` (proto)의 두 가지 출력을 가집니다.
            # 현재 ONNXRuntime 코드에서는 outputs[0]만 사용하고 있으므로, 이를 기반으로 합니다.
            # 여기서 outputs는 ONNXRuntime 코드에서 사용된 튜플/리스트 형태를 모방합니다.
            # 따라서, main_output_info["data"]를 직접 사용하는 대신, outputs[0]을 넘겨주는 방식으로 변경합니다.

            boxes, scores, labels, masks = self.postprocess(
                trt_engine_outputs, original_height=original_h, original_width=original_w, scale=scale_factor, padding=pad_offsets
            )

            detections = []
            for i in range(len(boxes)):
                x1, y1, x2, y2 = boxes[i]
                conf = scores[i]
                cls_id = labels[i]
                class_name = self.class_names[cls_id] if 0 <= cls_id < len(self.class_names) else str(cls_id)
                
                polygons = [] # 현재 마스크 처리가 없으므로 비어 있음
                # if masks and len(masks) > i: # 마스크가 있다면
                #     mask = masks[i]
                #     contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                #     for cnt in contours:
                #         polygon = [(int(x), int(y)) for [[x, y]] in cnt]
                #         if len(polygon) > 2:
                #             polygons.append(polygon)

                detections.append({
                    'box': (float(x1), float(y1), float(x2), float(y2)),
                    'conf': float(conf),
                    'class_name': class_name,
                    'polygons': polygons
                })
            return detections
        finally:
            # detect_objects가 끝나면 컨텍스트 비활성화
            if hasattr(self, 'cuda_context') and self.cuda_context:
                self.cuda_context.pop()
                # print("DEBUG: CUDA Context popped in detect_objects.") # 디버깅용


    def postprocess(self, outputs, original_width, original_height, scale, padding):
        # TensorRT 엔진의 출력 형태에 따라 이 부분을 조정해야 합니다.
        # ONNXRuntime 코드에서는 outputs[0].squeeze(0)이었으므로, 이를 반영합니다.
        # TensorRT 엔진 출력이 이미 배치 차원 1이 제거된 형태라면 squeeze(0)는 불필요할 수 있습니다.
        pred_raw = outputs[0].squeeze(0) # TensorRT 출력도 (1, N_outputs, N_proposals) 또는 (1, N_proposals, N_outputs) 형태일 수 있음
        
        # ONNXRuntime 코드와 동일하게 pred_raw.T를 수행
        pred = pred_raw.T
        
        boxes_raw = pred[:, 0:4]
        class_scores_logits_raw = pred[:, 4 : 4 + self.num_classes]
        # mask_coeffs_raw = pred[:, 4 + self.num_classes :] # 현재 마스크 사용 안 함

        # 시그모이드 적용
        class_scores_np = self.sigmoid(class_scores_logits_raw)
        
        scores_max = np.max(class_scores_np, axis=1)
        labels = np.argmax(class_scores_np, axis=1)
        
        keep_mask = scores_max > self.conf_threshold
        if keep_mask.sum() == 0:
            print(f"DEBUG: No objects after confidence threshold filtering (scores_max too low or conf_threshold {self.conf_threshold} too high).")
            return np.array([]), np.array([]), np.array([]), np.array([])
            
        boxes_filtered = boxes_raw[keep_mask]
        scores_filtered = scores_max[keep_mask]
        labels_filtered = labels[keep_mask]
        # mask_coeffs_filtered = mask_coeffs_raw[keep_mask] # 현재 마스크 사용 안 함
        
        # NMS를 위해 xywh를 xyxy로 변환
        boxes_xyxy = np.copy(boxes_filtered)
        boxes_xyxy[:, 0] = boxes_filtered[:, 0] - boxes_filtered[:, 2] / 2
        boxes_xyxy[:, 1] = boxes_filtered[:, 1] - boxes_filtered[:, 3] / 2
        boxes_xyxy[:, 2] = boxes_filtered[:, 0] + boxes_filtered[:, 2] / 2
        boxes_xyxy[:, 3] = boxes_filtered[:, 1] + boxes_filtered[:, 3] / 2
        
        # NMSBoxes는 x,y,w,h 형태의 박스를 기대합니다.
        boxes_for_nms = np.array([[x1, y1, x2-x1, y2-y1] for x1, y1, x2, y2 in boxes_xyxy])
        
        indices = cv2.dnn.NMSBoxes(boxes_for_nms.tolist(), scores_filtered.tolist(), self.conf_threshold, self.iou_threshold)
        
        if len(indices) == 0:
            print("DEBUG: No objects after NMS.")
            return np.array([]), np.array([]), np.array([]), np.array([])
            
        indices = indices.flatten()

        boxes_final = boxes_xyxy[indices]
        scores_final = scores_filtered[indices]
        labels_final = labels_filtered[indices]
        # mask_coeffs_final = mask_coeffs_filtered[indices] # 현재 마스크 사용 안 함
        
        # 패딩 및 스케일 역변환
        pad_left, pad_top = padding # padding은 (pad_left, pad_top) 형태로 전달됨 (preprocess에서 수정)

        # 1. 패딩 역변환
        boxes_final[:, 0] -= pad_left
        boxes_final[:, 2] -= pad_left
        boxes_final[:, 1] -= pad_top
        boxes_final[:, 3] -= pad_top

        # 2. 스케일 역변환
        boxes_final /= scale

        # 3. 클리핑하여 원본 이미지 경계 내에 있도록 합니다.
        boxes_final[:, 0] = np.clip(boxes_final[:, 0], 0, original_width)
        boxes_final[:, 1] = np.clip(boxes_final[:, 1], 0, original_height)
        boxes_final[:, 2] = np.clip(boxes_final[:, 2], 0, original_width)
        boxes_final[:, 3] = np.clip(boxes_final[:, 3], 0, original_height)
        
        # 유효한 박스 크기 보장 (최소 1픽셀)
        boxes_final[:, 2] = np.maximum(boxes_final[:, 2], boxes_final[:, 0] + 1)
        boxes_final[:, 3] = np.maximum(boxes_final[:, 3], boxes_final[:, 1] + 1)

        final_masks = [] # 현재 마스크 처리 없음
        
        return boxes_final, scores_final, labels_final, final_masks

    def __del__(self):
        # 모든 GPU 메모리 해제
        if hasattr(self, 'input_device_mem') and self.input_device_mem is not None:
            try:
                self.input_device_mem.free()
                self.input_device_mem = None
            except Exception as e:
                print(f"DEBUG: 입력 GPU 메모리 해제 중 오류: {e}")
        
        if hasattr(self, 'output_device_mem'):
            for name, dev_mem in self.output_device_mem.items():
                if dev_mem is not None:
                    try:
                        dev_mem.free()
                    except Exception as e:
                        print(f"DEBUG: 출력 GPU 메모리 '{name}' 해제 중 오류: {e}")
            self.output_device_mem = {}

        # TensorRT 컨텍스트 및 엔진 객체 정리
        if hasattr(self, 'context') and self.context is not None:
            del self.context
            self.context = None
        if hasattr(self, 'engine') and self.engine is not None:
            del self.engine
            self.engine = None
        
        # PyCUDA 컨텍스트는 primary context이므로 명시적으로 detach()하지 않는 것이 좋습니다.
        # 파이썬 프로그램 종료 시 자동으로 정리됩니다.
        # if hasattr(self, 'cuda_context') and self.cuda_context:
        #     self.cuda_context.detach() # 다른 스레드에서 생성된 컨텍스트는 detach() 필요
        # self.cuda_context = None

        print("DEBUG: Detector 소멸됨.")