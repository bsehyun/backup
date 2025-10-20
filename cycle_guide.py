import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

# ============================================================================
# 1. Feature별 최적 범위 및 가이드 정의
# ============================================================================

GOLDEN_FEATURE_SPECS = {
    'phase1': {
        'phase1_trend': {
            'optimal_range': (0.30, 0.40),
            'unit': '°C/min',
            'importance': 'Critical',
            'guidance_template': {
                'too_low': '⚠️ 온도 상승이 너무 느립니다 (목표: 0.30~0.40°C/min)\n→ 가열기 출력 증가 또는 반응 용기 단열 개선',
                'too_high': '⚠️ 온도 상승이 너무 빠릅니다 (목표: 0.30~0.40°C/min)\n→ 가열기 출력 감소 또는 냉각수 유량 증가',
                'optimal': '✓ 온도 상승률이 최적 범위입니다. 현재 상태 유지',
                'trend_info': '현재 추세: {current:.3f}°C/min (목표: {optimal_min:.3f}~{optimal_max:.3f}°C/min, 차이: {diff:.3f})'
            },
            'action_priority': 1
        },
        'phase1_acceleration': {
            'optimal_range': (0.001, 0.005),
            'unit': '°C/min²',
            'importance': 'High',
            'guidance_template': {
                'too_low': '△ 온도 상승 가속도가 낮습니다 (불안정한 상승)\n→ 가열 설정 안정화, 온도 센서 확인',
                'too_high': '△ 온도 상승 가속도가 높습니다 (급격한 변화)\n→ 가열 출력 평탄화, PID 제어 튜닝',
                'optimal': '✓ 온도 상승이 안정적입니다',
                'trend_info': '현재 가속도: {current:.5f}°C/min² (목표: {optimal_min:.5f}~{optimal_max:.5f}°C/min²)'
            },
            'action_priority': 2
        },
        'phase1_stability': {
            'optimal_range': (0.0, 0.15),
            'unit': 'std',
            'importance': 'Medium',
            'guidance_template': {
                'too_high': '⚠️ 온도 변동성이 높습니다 (불안정)\n→ 냉각/가열 시스템 확인, 센서 노이즈 제거',
                'optimal': '✓ 온도가 안정적으로 상승합니다',
                'trend_info': '현재 표준편차: {current:.4f} (목표: {optimal_max:.4f} 이하)'
            },
            'action_priority': 3
        }
    },
    'phase2': {
        'phase2_oscillation_amp': {
            'optimal_range': (4.5, 6.0),
            'unit': 'mV',
            'importance': 'Critical',
            'guidance_template': {
                'too_low': '⚠️ 오실레이션 폭이 너무 작습니다 (반응 부족)\n→ 촉매 농도 확인, 반응 온도 상향, 교반 속도 증가',
                'too_high': '⚠️ 오실레이션 폭이 너무 큽니다 (폭주 위험)\n→ 냉각 시스템 강화, 촉매 농도 감소, 반응 온도 하향',
                'optimal': '✓ 오실레이션이 최적 범위입니다. 현재 상태 유지',
                'trend_info': '현재 오실레이션: {current:.2f}mV (목표: {optimal_min:.2f}~{optimal_max:.2f}mV, 안정성: {stability})'
            },
            'action_priority': 1
        },
        'phase2_oscillation_frequency': {
            'optimal_range': (0.02, 0.05),
            'unit': '1/sec',
            'importance': 'High',
            'guidance_template': {
                'too_low': '△ 오실레이션 주기가 길어지고 있습니다 (반응 저하)\n→ 온도 상향, 촉매 활성 확인, 혼합 상태 개선',
                'too_high': '△ 오실레이션 주기가 짧아지고 있습니다 (과반응)\n→ 온도 하향, 공급 속도 감소, 냉각 강화',
                'optimal': '✓ 오실레이션 주기가 안정적입니다',
                'trend_info': '현재 주파수: {current:.4f}Hz (목표: {optimal_min:.4f}~{optimal_max:.4f}Hz)'
            },
            'action_priority': 2
        },
        'phase2_peak_consistency': {
            'optimal_range': (0.0, 0.3),
            'unit': 'variance',
            'importance': 'High',
            'guidance_template': {
                'too_high': '⚠️ 오실레이션 피크가 불일정합니다 (반응 조건 변화)\n→ 온도 제어 강화, 공급 펌프 정상 확인, 촉매 활성도 재평가',
                'optimal': '✓ 오실레이션이 일정한 패턴을 유지합니다',
                'trend_info': '현재 피크 분산: {current:.4f} (목표: {optimal_max:.4f} 이하)'
            },
            'action_priority': 2
        },
        'phase2_damping': {
            'optimal_range': (0.01, 0.05),
            'unit': 'damping_ratio',
            'importance': 'High',
            'guidance_template': {
                'too_low': '△ 오실레이션이 수렴하지 않고 지속됩니다 (시스템 게인 과다)\n→ 냉각/가열 PID 제어 튜닝, 시스템 응답성 저하',
                'too_high': '△ 오실레이션이 빠르게 감소합니다 (반응 종료 신호)\n→ 공급 속도 확인, 원료 고갈 여부 점검',
                'optimal': '✓ 오실레이션이 적절하게 감소/유지됩니다',
                'trend_info': '현재 감쇠율: {current:.4f} (목표: {optimal_min:.4f}~{optimal_max:.4f})'
            },
            'action_priority': 3
        }
    },
    'phase3': {
        'phase3_trend': {
            'optimal_range': (-0.25, -0.15),
            'unit': '°C/min',
            'importance': 'High',
            'guidance_template': {
                'too_high': '⚠️ 온도 감소 속도가 너무 느립니다 (냉각 부족)\n→ 냉각수 유량 증가, 냉각기 효율 확인, 환기 강화',
                'too_low': '⚠️ 온도 감소 속도가 너무 빠릅니다 (과냉각)\n→ 냉각수 유량 감소, 냉각기 설정 하향',
                'optimal': '✓ 온도가 최적 속도로 냉각 중입니다',
                'trend_info': '현재 냉각률: {current:.3f}°C/min (목표: {optimal_min:.3f}~{optimal_max:.3f}°C/min)'
            },
            'action_priority': 1
        },
        'phase3_stability': {
            'optimal_range': (0.0, 0.10),
            'unit': 'std',
            'importance': 'Medium',
            'guidance_template': {
                'too_high': '⚠️ 냉각 중 온도 진동이 있습니다 (냉각 제어 불안정)\n→ 냉각 PID 제어 튜닝, 냉각수 온도 안정화',
                'optimal': '✓ 냉각이 안정적으로 진행 중입니다',
                'trend_info': '현재 표준편차: {current:.4f} (목표: {optimal_max:.4f} 이하)'
            },
            'action_priority': 2
        },
        'phase3_final_temp': {
            'optimal_range': (25.0, 35.0),
            'unit': '°C',
            'importance': 'High',
            'guidance_template': {
                'too_high': '⚠️ 최종 온도가 목표보다 높습니다 (미냉각)\n→ 냉각 시간 연장, 냉각수 온도 하향',
                'too_low': '⚠️ 최종 온도가 목표보다 낮습니다 (과냉각)\n→ 냉각 종료 시간 단축, 냉각수 온도 상향',
                'optimal': '✓ 최종 온도가 목표 범위에 도달했습니다',
                'trend_info': '현재 온도: {current:.1f}°C (목표: {optimal_min:.1f}~{optimal_max:.1f}°C)'
            },
            'action_priority': 2
        }
    }
}

# ============================================================================
# 2. Phase별 운영 가이드 클래스
# ============================================================================

class OperationalGuide:
    """
    ACN 정제 배치의 실시간 운영 가이드 제공
    """
    
    def __init__(self, golden_specs=GOLDEN_FEATURE_SPECS):
        self.specs = golden_specs
        self.history = []
        self.alerts = []
    
    def evaluate_feature(self, feature_name: str, current_value: float, 
                        phase: str) -> Dict:
        """
        단일 특성 평가 및 가이드 생성
        """
        if phase not in self.specs:
            return {'error': f'Unknown phase: {phase}'}
        
        if feature_name not in self.specs[phase]:
            return {'error': f'Unknown feature: {feature_name}'}
        
        spec = self.specs[phase][feature_name]
        optimal_min, optimal_max = spec['optimal_range']
        
        # 상태 판정
        if current_value < optimal_min:
            status = 'TOO_LOW'
            guidance_key = 'too_low'
            severity = 'WARNING' if spec['importance'] == 'Critical' else 'INFO'
        elif current_value > optimal_max:
            status = 'TOO_HIGH'
            guidance_key = 'too_high'
            severity = 'WARNING' if spec['importance'] == 'Critical' else 'INFO'
        else:
            status = 'OPTIMAL'
            guidance_key = 'optimal'
            severity = 'OK'
        
        # 가이드 텍스트 생성
        guidance_template = spec['guidance_template'].get(guidance_key, '')
        
        # 추가 정보
        diff = current_value - ((optimal_min + optimal_max) / 2)
        deviation_pct = (abs(diff) / ((optimal_max - optimal_min) / 2)) * 100
        
        trend_info = spec['guidance_template'].get('trend_info', '').format(
            current=current_value,
            optimal_min=optimal_min,
            optimal_max=optimal_max,
            diff=diff,
            stability='안정' if deviation_pct < 20 else '불안정' if deviation_pct > 50 else '주의'
        )
        
        return {
            'feature': feature_name,
            'phase': phase,
            'current_value': current_value,
            'optimal_range': (optimal_min, optimal_max),
            'unit': spec['unit'],
            'importance': spec['importance'],
            'status': status,
            'severity': severity,
            'deviation_pct': deviation_pct,
            'guidance': guidance_template,
            'trend_info': trend_info,
            'action_priority': spec['action_priority']
        }
    
    def generate_batch_report(self, batch_data: Dict) -> str:
        """
        배치 진행 중 실시간 운영 리포트 생성
        
        batch_data 형식:
        {
            'batch_id': 'B001',
            'current_phase': 'phase2',
            'phase1_features': {'phase1_trend': 0.35, 'phase1_stability': 0.12, ...},
            'phase2_features': {'phase2_oscillation_amp': 5.2, ...},
            'phase3_features': {...}
        }
        """
        batch_id = batch_data.get('batch_id', 'Unknown')
        current_phase = batch_data.get('current_phase', 'unknown')
        
        report = f"""
╔════════════════════════════════════════════════════════════════╗
║              ACN 정제 배치 운영 가이드 시스템                  ║
║              Batch ID: {batch_id:<45}║
║              현재 Phase: {current_phase.upper():<40}║
╚════════════════════════════════════════════════════════════════╝

"""
        
        all_evaluations = []
        
        # 모든 Phase의 특성 평가
        for phase_key in ['phase1', 'phase2', 'phase3']:
            phase_data = batch_data.get(f'{phase_key}_features', {})
            
            if not phase_data:
                continue
            
            for feature_name, current_value in phase_data.items():
                evaluation = self.evaluate_feature(feature_name, current_value, phase_key)
                if 'error' not in evaluation:
                    all_evaluations.append(evaluation)
        
        # 심각도별 정렬
        severity_order = {'WARNING': 0, 'INFO': 1, 'OK': 2}
        all_evaluations.sort(key=lambda x: (severity_order.get(x['severity'], 3), 
                                            x['action_priority']))
        
        # 섹션별 리포트
        critical_issues = [e for e in all_evaluations if e['severity'] == 'WARNING']
        info_issues = [e for e in all_evaluations if e['severity'] == 'INFO']
        optimal_items = [e for e in all_evaluations if e['status'] == 'OPTIMAL']
        
        # 1. 긴급 조치 필요 사항
        if critical_issues:
            report += "🚨 긴급 조치 필요 사항\n"
            report += "─" * 60 + "\n"
            for i, issue in enumerate(critical_issues, 1):
                report += f"\n[{i}] {issue['feature']} ({issue['phase'].upper()})\n"
                report += f"    현재값: {issue['current_value']:.4f} {issue['unit']}\n"
                report += f"    목표범위: {issue['optimal_range'][0]:.4f} ~ {issue['optimal_range'][1]:.4f}\n"
                report += f"    편차: {issue['deviation_pct']:.1f}%\n"
                report += f"    중요도: {issue['importance']}\n"
                report += f"    {issue['guidance']}\n"
                report += f"    → 즉시 조치 필요!\n"
        
        # 2. 주의 사항
        if info_issues:
            report += "\n\n⚠️  주의 사항\n"
            report += "─" * 60 + "\n"
            for i, issue in enumerate(info_issues, 1):
                report += f"\n[{i}] {issue['feature']} ({issue['phase'].upper()})\n"
                report += f"    현재값: {issue['current_value']:.4f} {issue['unit']}\n"
                report += f"    목표범위: {issue['optimal_range'][0]:.4f} ~ {issue['optimal_range'][1]:.4f}\n"
                report += f"    {issue['guidance']}\n"
        
        # 3. 최적 상태 유지 중인 항목
        if optimal_items:
            report += "\n\n✓ 최적 상태 유지 중인 항목\n"
            report += "─" * 60 + "\n"
            for issue in optimal_items:
                report += f"• {issue['feature']}: {issue['trend_info']}\n"
        
        # 4. 종합 의견
        report += "\n\n" + "=" * 60 + "\n"
        report += "📋 종합 의견\n"
        report += "=" * 60 + "\n"
        
        if critical_issues:
            report += f"⚠️ CRITICAL: 즉시 조치가 필요한 항목 {len(critical_issues)}개\n"
            report += f"   → 현재 배치의 수율이 저하될 위험이 있습니다.\n"
            report += f"   → 다음 조치를 우선 실행하세요:\n"
            
            for issue in critical_issues:
                if 'too_low' in issue['guidance'].lower():
                    report += f"      • {issue['feature']}: ↑ 값을 증가시키세요\n"
                elif 'too_high' in issue['guidance'].lower():
                    report += f"      • {issue['feature']}: ↓ 값을 감소시키세요\n"
        
        if info_issues:
            report += f"\n⚠️ INFO: 추가 주의가 필요한 항목 {len(info_issues)}개\n"
            report += f"   → 향후 추이를 모니터링하세요.\n"
        
        if not critical_issues and not info_issues:
            report += "✓ GREEN: 모든 항목이 최적 범위 내입니다!\n"
            report += "   → 현재 상태를 유지하세요.\n"
        
        report += "\n" + "=" * 60 + "\n"
        report += "📊 다음 점검 주기: 5분 후\n"
        report += "=" * 60 + "\n"
        
        return report
    
    def get_quick_action_list(self, batch_data: Dict) -> List[str]:
        """
        빠른 조치 목록 (운영자용 핵심 요약)
        """
        actions = []
        
        for phase_key in ['phase1', 'phase2', 'phase3']:
            phase_data = batch_data.get(f'{phase_key}_features', {})
            
            for feature_name, current_value in phase_data.items():
                evaluation = self.evaluate_feature(feature_name, current_value, phase_key)
                
                if 'error' in evaluation:
                    continue
                
                if evaluation['severity'] == 'WARNING':
                    if 'too_low' in evaluation['guidance']:
                        action = f"↑ [{phase_key.upper()}] {feature_name} 증가"
                    elif 'too_high' in evaluation['guidance']:
                        action = f"↓ [{phase_key.upper()}] {feature_name} 감소"
                    else:
                        action = f"⚡ [{phase_key.upper()}] {feature_name} 조정 필요"
                    
                    actions.append(action)
        
        return actions
    
    def compare_with_golden(self, current_batch: Dict, golden_reference: Dict) -> str:
        """
        현재 배치와 Golden cycle 비교
        """
        report = """
╔════════════════════════════════════════════════════════════════╗
║           현재 배치 vs Golden Cycle 비교                       ║
╚════════════════════════════════════════════════════════════════╝

"""
        
        comparison_data = []
        
        for phase_key in ['phase1', 'phase2', 'phase3']:
            current_features = current_batch.get(f'{phase_key}_features', {})
            golden_features = golden_reference.get(f'{phase_key}_features', {})
            
            report += f"\n{phase_key.upper()}\n"
            report += "─" * 60 + "\n"
            
            for feature_name, current_value in current_features.items():
                golden_value = golden_features.get(feature_name)
                
                if golden_value is None:
                    continue
                
                diff = current_value - golden_value
                diff_pct = (diff / abs(golden_value)) * 100 if golden_value != 0 else 0
                
                if abs(diff_pct) < 5:
                    indicator = "✓"
                    status = "일치"
                elif abs(diff_pct) < 15:
                    indicator = "△"
                    status = "약간 차이"
                else:
                    indicator = "✗"
                    status = "큰 차이"
                
                report += f"{indicator} {feature_name}\n"
                report += f"   현재: {current_value:.4f}  |  Golden: {golden_value:.4f}  |  차이: {diff_pct:+.1f}% ({status})\n"
        
        return report

# ============================================================================
# 3. 사용 예시
# ============================================================================

if __name__ == "__main__":
    # 운영 가이드 초기화
    guide = OperationalGuide()
    
    # 예시 배치 데이터 (진행 중)
    current_batch_phase2 = {
        'batch_id': 'B001_ACN_20251020',
        'current_phase': 'phase2',
        'phase1_features': {
            'phase1_trend': 0.32,
            'phase1_acceleration': 0.003,
            'phase1_stability': 0.08
        },
        'phase2_features': {
            'phase2_oscillation_amp': 7.2,  # 너무 높음
            'phase2_oscillation_frequency': 0.035,
            'phase2_peak_consistency': 0.25,
            'phase2_damping': 0.03
        },
        'phase3_features': {}
    }
    
    # 리포트 생성
    print(guide.generate_batch_report(current_batch_phase2))
    
    # 빠른 조치 목록
    print("\n\n🎯 빠른 조치 목록:\n")
    actions = guide.get_quick_action_list(current_batch_phase2)
    for i, action in enumerate(actions, 1):
        print(f"{i}. {action}")
    
    # Golden cycle 참조 데이터
    golden_reference = {
        'batch_id': 'GOLDEN_REFERENCE',
        'current_phase': 'phase2',
        'phase1_features': {
            'phase1_trend': 0.35,
            'phase1_acceleration': 0.004,
            'phase1_stability': 0.05
        },
        'phase2_features': {
            'phase2_oscillation_amp': 5.2,
            'phase2_oscillation_frequency': 0.038,
            'phase2_peak_consistency': 0.15,
            'phase2_damping': 0.025
        },
        'phase3_features': {}
    }
    
    # Golden 비교
    print(guide.compare_with_golden(current_batch_phase2, golden_reference))
