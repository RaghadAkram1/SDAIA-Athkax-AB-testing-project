"""Day 2 — An independently implemented power plan via statsmodels."""
import json
import math
from pathlib import Path
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportion_effectsize
from day_01_question_and_design.research_design import Study

BASE = Path(__file__).resolve().parents[1]

def run(study=None):
    study = study or Study()
    target = study.baseline_rate + study.meaningful_gain
    if not (0 < study.baseline_rate < 1 and 0 < target < 1):
        raise ValueError('يجب أن تقع النسب بين صفر وواحد')
    magnitude = abs(proportion_effectsize(target, study.baseline_rate))
    power_model = NormalIndPower()
    per_arm = math.ceil(power_model.solve_power(effect_size=magnitude,
                  alpha=study.alpha, power=study.target_power,
                  ratio=(1-study.allocation_to_ai)/study.allocation_to_ai,
                  alternative='two-sided'))
    target_total = math.ceil(per_arm/study.allocation_to_ai)
    necessary_days = math.ceil(target_total/study.new_users_daily)
    calendar_days = max(7*study.minimum_weeks, 7*math.ceil(necessary_days/7))
    expected_users = calendar_days * study.new_users_daily
    output = {
        'control_baseline_assumed': study.baseline_rate,
        'treatment_target_for_power': target,
        'minimum_detectable_absolute_effect': study.meaningful_gain,
        'alpha_two_sided': study.alpha,
        'power_goal': study.target_power,
        'minimum_per_arm_approx': per_arm,
        'minimum_total_approx': target_total,
        'eligible_per_day_assumed': study.new_users_daily,
        'days_needed_for_minimum': necessary_days,
        'scheduled_days_full_weeks': calendar_days,
        'expected_assigned_total': expected_users,
        'calculation': 'حساب تقريبي باستخدام Cohen h ونموذج NormalIndPower لنسبتين، مع افتراض استقلال الحسابات والتوزيع المتساوي وعدم فقد البيانات.'
    }
    location = BASE/'outputs/results/day_02_power_plan.json'
    location.parent.mkdir(parents=True, exist_ok=True)
    location.write_text(json.dumps(output, indent=2), encoding='utf-8')
    print(f'[اليوم الثاني] حجم العينة لكل مجموعة: {per_arm:,} مستخدم؛ مدة التجربة المخططة: {calendar_days} يومًا')
    return output

if __name__ == '__main__':
    run()
