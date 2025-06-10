"""
확장된 일괄 프루닝 시스템
기존 성공한 시스템에 더 많은 후보 레이어들을 추가하여 자동 탐색
"""

import torch
import torch.nn as nn
from ultralytics import YOLO
import copy
import time
import numpy as np

class ExpandedBatchPruner:
    def __init__(self, model_path):
        self.yolo = YOLO(model_path)
        self.model = self.yolo.model
        
        # 확장된 후보 레이어들 (기존 성공 + 새로운 후보들)
        self.target_candidates = [
            # 기존 성공한 cv1 레이어들 (검증됨)
            "model.2.m.0.cv1.conv",   # 16 채널
            "model.4.m.0.cv1.conv",   # 32 채널  
            "model.4.m.1.cv1.conv",   # 32 채널
            "model.6.m.0.cv1.conv",   # 64 채널
            "model.6.m.1.cv1.conv",   # 64 채널
            "model.8.m.0.cv1.conv",   # 128 채널
            "model.12.m.0.cv1.conv",  # 64 채널
            "model.15.m.0.cv1.conv",  # 32 채널
            "model.18.m.0.cv1.conv",  # 64 채널
            "model.21.m.0.cv1.conv",  # 128 채널
            
            # 새로운 후보: cv2 레이어들 (높은 성공 확률)
            "model.2.m.0.cv2.conv",   # 16 채널
            "model.4.m.0.cv2.conv",   # 32 채널
            "model.4.m.1.cv2.conv",   # 32 채널
            "model.6.m.0.cv2.conv",   # 64 채널
            "model.6.m.1.cv2.conv",   # 64 채널
            "model.8.m.0.cv2.conv",   # 128 채널
            "model.12.m.0.cv2.conv",  # 64 채널
            "model.15.m.0.cv2.conv",  # 32 채널
            "model.18.m.0.cv2.conv",  # 64 채널
            "model.21.m.0.cv2.conv",  # 128 채널
            
            # 새로운 후보: Backbone Conv 레이어들 (중간 확률)
            "model.1.conv",           # 32 채널
            "model.3.conv",           # 64 채널
            "model.5.conv",           # 128 채널
            "model.7.conv",           # 256 채널
            
            # 새로운 후보: 기타 레이어들 (낮은 확률)
            "model.16.conv",          # 64 채널
            "model.19.conv",          # 128 채널
            "model.9.cv1.conv",       # SPPF 128 채널
            "model.9.cv2.conv",       # SPPF 256 채널
        ]
        
        self.original_model = None
        
    def print_model_info(self, label="모델"):
        """모델 정보 출력"""
        total_params = sum(p.numel() for p in self.model.parameters())
        param_size = sum(param.nelement() * param.element_size() for param in self.model.parameters())
        buffer_size = sum(buffer.nelement() * buffer.element_size() for buffer in self.model.buffers())
        size_mb = (param_size + buffer_size) / 1024 / 1024
        
        print(f"📊 {label} 정보:")
        print(f"   파라미터 수: {total_params:,}개")
        print(f"   모델 크기: {size_mb:.2f} MB")
    
    def scan_available_candidates(self):
        """실제 존재하는 후보 레이어들 스캔 및 분류"""
        print("🔍 확장된 후보 레이어 스캔:")
        
        available_layers = []
        categories = {
            'cv1_existing': [],
            'cv2_new': [],
            'backbone_new': [],
            'other_new': []
        }
        
        for target in self.target_candidates:
            layer = None
            
            # 레이어 존재 확인
            for name, module in self.model.named_modules():
                if name == target and isinstance(module, nn.Conv2d):
                    layer = module
                    break
            
            if layer:
                available_layers.append((target, layer))
                
                # 카테고리 분류
                if 'cv1.conv' in target:
                    categories['cv1_existing'].append((target, layer))
                elif 'cv2.conv' in target:
                    categories['cv2_new'].append((target, layer))
                elif target in ['model.1.conv', 'model.3.conv', 'model.5.conv', 'model.7.conv']:
                    categories['backbone_new'].append((target, layer))
                else:
                    categories['other_new'].append((target, layer))
                
                print(f"   ✅ {target} ({layer.out_channels} 채널)")
            else:
                print(f"   ❌ {target} (없음)")
        
        # 카테고리별 요약
        print(f"\n📋 후보 분류:")
        print(f"   기존 cv1 (검증됨): {len(categories['cv1_existing'])}개")
        print(f"   새로운 cv2: {len(categories['cv2_new'])}개")
        print(f"   새로운 backbone: {len(categories['backbone_new'])}개")
        print(f"   기타 새로운: {len(categories['other_new'])}개")
        print(f"   총 후보: {len(available_layers)}개")
        
        return available_layers, categories
    
    def find_connected_conv_pair(self, layer_name):
        """연결된 Conv 쌍 찾기"""
        if 'cv1.conv' in layer_name:
            cv2_name = layer_name.replace('cv1.conv', 'cv2.conv')
        elif 'cv2.conv' in layer_name:
            cv1_name = layer_name.replace('cv2.conv', 'cv1.conv')
            return self.find_connected_conv_pair(cv1_name)
        else:
            return None, None, None
        
        cv1_layer = None
        cv2_layer = None
        
        for name, module in self.model.named_modules():
            if name == layer_name and isinstance(module, nn.Conv2d):
                cv1_layer = module
            elif name == cv2_name and isinstance(module, nn.Conv2d):
                cv2_layer = module
        
        if cv1_layer and cv2_layer:
            return cv1_layer, cv2_layer, cv2_name
        
        return None, None, None
    
    def find_conv_bn_pair(self, conv_name):
        """Conv와 BatchNorm 쌍 찾기"""
        bn_name = conv_name.replace('.conv', '.bn')
        
        conv_layer = None
        bn_layer = None
        
        for name, module in self.model.named_modules():
            if name == conv_name and isinstance(module, nn.Conv2d):
                conv_layer = module
            elif name == bn_name and isinstance(module, nn.BatchNorm2d):
                bn_layer = module
        
        return conv_layer, bn_layer
    
    def test_single_layer_safety(self, layer_name, prune_ratio=0.1):
        """단일 레이어 프루닝 안전성 테스트"""
        # 모델 백업
        original_model = copy.deepcopy(self.model)
        
        try:
            # 연결된 쌍이 있는지 확인
            cv1_layer, cv2_layer, cv2_name = self.find_connected_conv_pair(layer_name)
            
            if cv1_layer and cv2_layer:
                # cv1-cv2 쌍 프루닝
                success = self._prune_conv_pair(layer_name, cv1_layer, cv2_name, cv2_layer, prune_ratio)
            else:
                # 단독 레이어 프루닝
                success = self._prune_single_layer(layer_name, prune_ratio)
            
            if success:
                # 모델 동작 테스트
                dummy_input = torch.randn(1, 3, 192, 320)
                with torch.no_grad():
                    output = self.model(dummy_input)
                return True
            else:
                return False
                
        except Exception as e:
            print(f"      오류: {e}")
            return False
        
        finally:
            # 원본 모델 복원
            self.model = original_model
            self.yolo.model = self.model
    
    def _prune_conv_pair(self, cv1_name, cv1_layer, cv2_name, cv2_layer, prune_ratio):
        """CV1-CV2 연결 쌍 프루닝"""
        original_channels = cv1_layer.out_channels
        prune_count = int(original_channels * prune_ratio)
        prune_count = min(prune_count, original_channels - 8)
        
        if prune_count <= 0:
            return False
        
        # 중요도 계산
        weights = cv1_layer.weight.data
        channel_importance = torch.norm(weights.view(weights.size(0), -1), dim=1)
        _, prune_indices = torch.topk(channel_importance, prune_count, largest=False)
        keep_indices = torch.tensor([i for i in range(original_channels) if i not in prune_indices])
        
        # CV1 출력 채널 프루닝
        self._prune_conv_output_channels(cv1_layer, cv1_name, keep_indices)
        
        # CV2 입력 채널 프루닝
        self._prune_conv_input_channels(cv2_layer, keep_indices)
        
        return True
    
    def _prune_single_layer(self, layer_name, prune_ratio):
        """단독 레이어 프루닝"""
        conv_layer, bn_layer = self.find_conv_bn_pair(layer_name)
        
        if conv_layer is None:
            return False
        
        original_channels = conv_layer.out_channels
        prune_count = int(original_channels * prune_ratio)
        prune_count = min(prune_count, original_channels - 8)
        
        if prune_count <= 0:
            return False
        
        weights = conv_layer.weight.data
        channel_importance = torch.norm(weights.view(weights.size(0), -1), dim=1)
        _, prune_indices = torch.topk(channel_importance, prune_count, largest=False)
        keep_indices = torch.tensor([i for i in range(original_channels) if i not in prune_indices])
        
        self._prune_conv_output_channels(conv_layer, layer_name, keep_indices)
        return True
    
    def _prune_conv_output_channels(self, conv_layer, layer_name, keep_indices):
        """Conv 출력 채널 프루닝"""
        # Conv 프루닝
        new_weight = conv_layer.weight.data[keep_indices]
        new_bias = conv_layer.bias.data[keep_indices] if conv_layer.bias is not None else None
        
        conv_layer.out_channels = len(keep_indices)
        conv_layer.weight = nn.Parameter(new_weight)
        if new_bias is not None:
            conv_layer.bias = nn.Parameter(new_bias)
        
        # BatchNorm 프루닝
        bn_name = layer_name.replace('.conv', '.bn')
        for name, module in self.model.named_modules():
            if name == bn_name and isinstance(module, nn.BatchNorm2d):
                self._prune_batchnorm(module, keep_indices)
                break
    
    def _prune_conv_input_channels(self, conv_layer, keep_indices):
        """Conv 입력 채널 프루닝"""
        new_weight = conv_layer.weight.data[:, keep_indices]
        conv_layer.in_channels = len(keep_indices)
        conv_layer.weight = nn.Parameter(new_weight)
    
    def _prune_batchnorm(self, bn_layer, keep_indices):
        """BatchNorm 프루닝"""
        bn_layer.num_features = len(keep_indices)
        
        if bn_layer.weight is not None:
            bn_layer.weight = nn.Parameter(bn_layer.weight.data[keep_indices])
        if bn_layer.bias is not None:
            bn_layer.bias = nn.Parameter(bn_layer.bias.data[keep_indices])
        
        bn_layer.running_mean = bn_layer.running_mean.data[keep_indices]
        bn_layer.running_var = bn_layer.running_var.data[keep_indices]
    
    def discover_safe_layers(self, test_ratio=0.1):
        """안전한 프루닝 레이어 자동 발견"""
        print(f"\n🔍 안전한 레이어 자동 탐색 (테스트 비율: {test_ratio:.0%})")
        print("=" * 60)
        
        # 후보 스캔
        available_layers, categories = self.scan_available_candidates()
        
        if not available_layers:
            print("❌ 테스트할 후보가 없습니다.")
            return []
        
        # 각 후보별 안전성 테스트
        safe_layers = []
        
        print(f"\n🧪 후보별 안전성 테스트:")
        for i, (layer_name, layer_module) in enumerate(available_layers, 1):
            print(f"\n--- {i}/{len(available_layers)}: {layer_name} ---")
            print(f"   채널 수: {layer_module.out_channels}")
            
            # 안전성 테스트
            if self.test_single_layer_safety(layer_name, test_ratio):
                safe_layers.append((layer_name, layer_module))
                print(f"   ✅ 안전 확인")
            else:
                print(f"   ❌ 위험 - 제외")
        
        # 결과 요약
        print(f"\n📊 탐색 결과:")
        print(f"   테스트한 후보: {len(available_layers)}개")
        print(f"   안전한 레이어: {len(safe_layers)}개")
        print(f"   성공률: {len(safe_layers)/len(available_layers)*100:.1f}%")
        
        if safe_layers:
            print(f"\n✅ 발견된 안전한 레이어들:")
            for layer_name, layer_module in safe_layers:
                print(f"   {layer_name} ({layer_module.out_channels} 채널)")
        
        return safe_layers
    
    def backup_model(self):
        """모델 백업"""
        self.original_model = copy.deepcopy(self.model)
        print("💾 원본 모델 백업 완료")
    
    def restore_model(self):
        """모델 복원"""
        if self.original_model:
            self.model = self.original_model
            self.yolo.model = self.model
            print("🔄 원본 모델 복원 완료")
    
    def benchmark_speed(self, num_runs=50):
        """추론 속도 벤치마크"""
        dummy_image = np.random.randint(0, 255, (192, 320, 3), dtype=np.uint8)
        
        # 워밍업
        for _ in range(5):
            _ = self.yolo.predict(dummy_image, verbose=False)
        
        # 벤치마크
        start_time = time.perf_counter()
        for _ in range(num_runs):
            _ = self.yolo.predict(dummy_image, verbose=False)
        end_time = time.perf_counter()
        
        avg_time = (end_time - start_time) / num_runs * 1000
        fps = 1000 / avg_time
        
        print(f"   추론 속도: {avg_time:.2f}ms ({fps:.1f} FPS)")
        return avg_time, fps
    
    def save_model(self, filename="expanded_pruned_model.pt"):
        """프루닝된 모델 저장"""
        save_path = f"./{filename}"
        self.yolo.save(save_path)
        print(f"💾 확장 프루닝된 모델 저장: {save_path}")
        return save_path


def main():
    MODEL_PATH = r"C:\Users\KDT34\Desktop\Group6\best.pt"
    TEST_RATIO = 0.1    # 10% 테스트 프루닝
    FINAL_RATIO = 0.15  # 15% 실제 프루닝
    
    print("🚀 확장된 일괄 프루닝 시스템")
    print("=" * 50)
    
    try:
        # 프루너 초기화
        pruner = ExpandedBatchPruner(MODEL_PATH)
        
        # 원본 성능 측정
        print("\n1️⃣ 원본 모델 성능:")
        pruner.print_model_info("원본")
        original_speed = pruner.benchmark_speed()
        
        # 안전한 레이어 자동 발견
        print("\n2️⃣ 안전한 레이어 자동 발견:")
        safe_layers = pruner.discover_safe_layers(TEST_RATIO)
        
        if not safe_layers:
            print("❌ 추가로 안전한 레이어를 찾지 못했습니다.")
            return
        
        # 사용자 확인
        print(f"\n발견된 {len(safe_layers)}개 레이어로 실제 프루닝을 진행하시겠습니까?")
        user_input = input("계속 진행? (y/n) [y]: ").strip().lower()
        if user_input == 'n':
            print("프로그램을 종료합니다.")
            return
        
        # 실제 프루닝 실행 (기존 일괄 프루닝 시스템 재사용)
        print(f"\n3️⃣ 확장된 일괄 프루닝 실행:")
        pruner.backup_model()
        
        # 발견된 안전한 레이어들로 타겟 리스트 업데이트
        pruner.target_candidates = [name for name, _ in safe_layers]
        
        # 기존 일괄 프루닝 로직 재사용
        success_count = 0
        total_channels_before = 0
        total_channels_after = 0
        
        for i, (layer_name, layer_module) in enumerate(safe_layers, 1):
            print(f"\n--- {i}/{len(safe_layers)}: {layer_name} ---")
            
            # 연결된 쌍 확인
            cv1_layer, cv2_layer, cv2_name = pruner.find_connected_conv_pair(layer_name)
            
            original_channels = layer_module.out_channels
            
            if cv1_layer and cv2_layer:
                # 쌍 프루닝
                success = pruner._prune_conv_pair(layer_name, cv1_layer, cv2_name, cv2_layer, FINAL_RATIO)
            else:
                # 단독 프루닝  
                success = pruner._prune_single_layer(layer_name, FINAL_RATIO)
            
            if success:
                after_channels = int(original_channels * (1 - FINAL_RATIO))
                after_channels = max(after_channels, 8)  # 최소 8채널
                
                print(f"   📊 채널 수: {original_channels} → {after_channels} ({original_channels-after_channels}개 제거)")
                total_channels_before += original_channels
                total_channels_after += after_channels
                
                # 모델 동작 확인
                try:
                    dummy_input = torch.randn(1, 3, 192, 320)
                    with torch.no_grad():
                        output = pruner.model(dummy_input)
                    success_count += 1
                    print(f"   ✅ 프루닝 성공")
                except Exception as e:
                    print(f"   ❌ 모델 오류 발생: {e}")
                    pruner.restore_model()
                    break
            else:
                print(f"   ❌ 프루닝 실패")
        
        # 결과 평가
        print(f"\n4️⃣ 확장 프루닝 결과:")
        if success_count > 0:
            print(f"   성공한 레이어: {success_count}/{len(safe_layers)}개")
            print(f"   총 채널 감소: {total_channels_before} → {total_channels_after}")
            print(f"   채널 감소율: {(1-total_channels_after/total_channels_before)*100:.1f}%")
            
            # 최종 성능 측정
            print(f"\n5️⃣ 최종 성능:")
            pruner.print_model_info("확장 프루닝")
            final_speed = pruner.benchmark_speed()
            
            # 성능 비교
            speed_improvement = original_speed[0] / final_speed[0]
            print(f"\n📊 성능 비교:")
            print(f"   원본: {original_speed[1]:.1f} FPS")
            print(f"   확장 프루닝 후: {final_speed[1]:.1f} FPS")
            print(f"   속도 향상: {speed_improvement:.2f}배")
            
            # 모델 저장
            save_choice = input("\n확장 프루닝된 모델을 저장하시겠습니까? (y/n) [y]: ").strip().lower()
            if save_choice != 'n':
                saved_path = pruner.save_model()
                print(f"\n✅ 확장 프루닝 완료!")
                print(f"최종 모델: {saved_path}")
        else:
            print("   ❌ 확장 프루닝에 실패했습니다.")
    
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()