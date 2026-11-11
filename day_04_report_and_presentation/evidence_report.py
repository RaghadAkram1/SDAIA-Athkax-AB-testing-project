# Day 4: turn experiment estimates into an evidence-based Arabic report.
from pathlib import Path
import json
from day_01_question_and_design.research_design import Study

ROOT = Path(__file__).resolve().parents[1]


def percentage_points(value):
    return f'{100 * value:+.2f} نقطة مئوية'


def decision_from_rules(study, audit, results):
    primary = results['primary']
    if not audit['data_usable_for_inference']:
        return 'إيقاف التقييم: فحوص جودة البيانات أو التوزيع لم تنجح'
    if primary['ci95_upper'] < 0:
        return 'عدم الإطلاق: توجد أدلة على تراجع نسبة الإكمال'
    guardrails_ok = all(results[key]['ci95_upper'] < study.safety_margin for key in ['errors', 'support'])
    if not guardrails_ok:
        return 'تعليق قرار الإطلاق: لا تكفي الأدلة لاستبعاد تدهور مهم في مقاييس الحماية'
    if primary['ci95_lower'] > 0 and primary['difference'] >= study.meaningful_gain:
        return 'النظر في إطلاق مشروط ضمن المحاكاة، بعد التحقق بتجربة فعلية'
    if primary['ci95_upper'] < study.meaningful_gain:
        return 'عدم الإطلاق حاليًا: البيانات لا تدعم أثرًا بالحجم العملي المحدد'
    return 'النتيجة غير حاسمة: نوصي بتجربة جديدة أكبر مخططة مسبقًا'


def run(study, plan, checks, results):
    output = ROOT / 'outputs/results'
    output.mkdir(parents=True, exist_ok=True)
    primary = results['primary']
    adjusted = results['ancova']
    recommendation = decision_from_rules(study, checks, results)
    safety_rows = []
    for name, key in [('حدوث أخطاء', 'errors'), ('التواصل مع الدعم', 'support')]:
        result = results[key]
        state = 'مقبول' if result['ci95_upper'] < study.safety_margin else 'غير محسوم'
        safety_rows.append(
            f"| {name} | {result['control_rate']:.2%} | {result['ai_rate']:.2%} | "
            f"{percentage_points(result['difference'])} | "
            f"{percentage_points(result['ci95_lower'])} إلى {percentage_points(result['ci95_upper'])} | {state} |"
        )
    report = f'''# اليوم الرابع | التقرير النهائي والتوصية

**اسم الدورة:** التجارب واختبارات A/B والاستدلال السببي — أكاديمية سدايا  
**اسم الدراسة:** {study.title}  
**مصدر الأدلة:** بيانات **محاكاة تعليمية** بالكامل، وليست نتائج على منصة حقيقية  
**التوصية النهائية:** **{recommendation}**

## السؤال والقرار
هل نضيف مساعدًا ذكيًا اختياريًا داخل خدمة إلكترونية افتراضية؟ قارنا المجموعتين A وB عبر توزيع عشوائي ثابت على مستوى الحساب، ودرسنا **أثر إتاحة** المساعد لا أثر استخدامه فقط. تعتمد النتيجة الأساسية على تحليل ITT لجميع المستخدمين الموزعين.

## التصميم وحجم العينة
- خط الأساس المفترض للإكمال: **{study.baseline_rate:.0%}**.
- أصغر أثر عملي مؤثر في القرار: **{percentage_points(study.meaningful_gain)}**.
- مستوى الدلالة: **{study.alpha:.0%}**؛ القوة: **{study.target_power:.0%}**؛ اختبار ثنائي الطرف.
- العينة المخططة: **{plan['minimum_per_arm_approx']:,}** حسابًا لكل مجموعة (تقريبًا).
- المدة: **{plan['scheduled_days_full_weeks']} يومًا**، بمتوسط **{study.new_users_daily:,}** حساب مؤهل يوميًا، أي **{plan['expected_assigned_total']:,}** مستخدم متوقع.

## فحوص جودة البيانات
- إجمالي السجلات: **{checks['users']:,}**، منها A = **{checks['A']:,}** وB = **{checks['B']:,}**.
- فحص SRM باستخدام مربع كاي: قيمة p = **{checks['srm_p_value']:.4f}**، وحد التنبيه **0.001**.
- السجلات المكررة: **{checks['duplicate_accounts']}**؛ عدد الأعمدة التي تضم قيماً إلزامية مفقودة: **{len(checks['missing_required'])}**.
- عدم تسجيل زمن إكمال للذين لم يكملوا متوقع؛ ولا يعني فقدًا في نتيجة الإكمال الأساسية.

## النتيجة الأساسية: إكمال الخدمة خلال 24 ساعة
- المجموعة A: **{primary['control_rate']:.2%}**.
- المجموعة B: **{primary['ai_rate']:.2%}**.
- فرق النسب (B−A) وفق ITT: **{percentage_points(primary['difference'])}**.
- فترة الثقة 95% بطريقة Newcombe–Wilson: **[{percentage_points(primary['ci95_lower'])}، {percentage_points(primary['ci95_upper'])}]**.
- قيمة p لاختبار النسبتين: **{primary['p_two_sided']:.6f}**.
- تحليل الحساسية ANCOVA مع ضبط الزيارات السابقة وأخطاء معيارية HC3: **{percentage_points(adjusted['difference'])}**؛ فترة الثقة 95% **[{percentage_points(adjusted['ci95_lower'])}، {percentage_points(adjusted['ci95_upper'])}]**. يبقى ITT غير المعدّل أساس اتخاذ القرار.

## مقاييس الحماية
شرط الحماية المحدد مسبقًا: الحد الأعلى لفترة الثقة في فرق الضرر لكل مقياس يجب أن يكون **أقل من +2 نقطة مئوية**. عدم العثور على دلالة للضرر لا يعني وحده إثبات السلامة.

| المقياس | المجموعة A | المجموعة B | الفرق B−A | فترة الثقة 95% | الحكم |
|---|---:|---:|---:|---|---|
{chr(10).join(safety_rows)}

## التفاعل مع الميزة والأهمية العملية
- فتح المساعد **{results['ai_widget_opened_among_B']:.1%}** من مستخدمي B. هذا قياس وصفي، وليس مقارنة سببية بين من فتحوا المساعد ومن لم يفتحوه.
- إذا افترضنا **100,000** مستخدم مماثل في ظروف المحاكاة نفسها، فإن التقدير النقطي يعادل تقريبًا **{primary['difference'] * 100000:,.0f}** إكمالًا إضافيًا (فترة الثقة من **{primary['ci95_lower'] * 100000:,.0f}** إلى **{primary['ci95_upper'] * 100000:,.0f}**). هذا **توضيح حسابي** وليس توقعًا فعليًا لمؤسسة حقيقية.
- زمن الإكمال لدى **من أكملوا الخدمة فقط**: A = **{results['minutes_among_completers_only']['A']:.1f} دقيقة** وB = **{results['minutes_among_completers_only']['B']:.1f} دقيقة**. المقارنة **وصفية وليست سببية** لأن المعالجة قد تغير من يستطيع الإكمال.

## تفسير النتائج والتوصية
**{recommendation}**. الدلالة الإحصائية تختلف عن الأهمية العملية؛ لذلك لا نعتمد قيمة p وحدها لاتخاذ قرار الإطلاق. الخطوة التالية في الواقع هي تجربة ميدانية مخططة تتضمن مراقبة جودة إجابات المساعد وسلامة البيانات والخصوصية وتكلفة التشغيل وإمكانية الوصول.

## القيود
كل الحسابات والنتائج وحجم الأثر وافتراضات سلوك المستخدمين **مصطنعة لأغراض التدريب**. لا تثبت هذه الدراسة أن المساعد ينجح في بيئة فعلية. لا تلتقط المحاكاة الهلوسة، والأخطاء التشغيلية الحقيقية، واعتبارات الخصوصية، أو الآثار طويلة المدى. بذرة الأرقام الثابتة تضمن إعادة إنتاج النتائج ولا تعني تحققها خارج المحاكاة.
'''
    (output / 'day_04_final_memo.md').write_text(report, encoding='utf-8')
    (output / 'day_04_recommendation.json').write_text(json.dumps({
        'decision': recommendation,
        'primary': primary,
        'guardrails': {name: results[name] for name in ['errors', 'support']},
        'synthetic_data': True,
    }, indent=2, ensure_ascii=False), encoding='utf-8')
    presentation = f'''# اليوم الرابع | نقاط عرض المشروع خلال 10 دقائق

1. **الدقيقة الأولى — المشكلة:** ما المشكلة التي قد يواجهها المستخدم عند إكمال خدمة إلكترونية، وما قرار المنتج الذي ندرسه؟
2. **الدقيقة الثانية — السؤال:** هل تزيد إتاحة مساعد ذكي نسبة الإكمال خلال 24 ساعة؟ اذكري الفرضية الصفرية والبديلة.
3. **الدقيقة الثالثة — التصميم السببي:** توزيع الحسابات عشوائيًا بالتساوي؛ A دون مساعد وB مع مساعد اختياري؛ تجنب تغيير مجموعة المستخدم عند تكرار الزيارة.
4. **الدقيقة الرابعة — المقاييس:** النسبة الأساسية للإكمال لجميع المستخدمين (ITT)، مع الأخطاء والدعم كمقياسي حماية.
5. **الدقيقة الخامسة — القوة:** العينة المخططة تقريبًا {plan['minimum_per_arm_approx']:,} لكل مجموعة، والمدة {plan['scheduled_days_full_weeks']} يومًا، والحد العملي +5 نقاط مئوية.
6. **الدقيقة السادسة — جودة البيانات:** لا سجلات مكررة، مع تحقق SRM وقيمة p = {checks['srm_p_value']:.4f}.
7. **الدقيقة السابعة — النتيجة:** A = {primary['control_rate']:.2%}، B = {primary['ai_rate']:.2%}، والأثر = {percentage_points(primary['difference'])}.
8. **الدقيقة الثامنة — عدم اليقين:** فترة الثقة 95% من {percentage_points(primary['ci95_lower'])} إلى {percentage_points(primary['ci95_upper'])}، وقيمة p = {primary['p_two_sided']:.5f}، مع تحليل ANCOVA إضافي.
9. **الدقيقة التاسعة — الحماية والأهمية العملية:** هل تدهورت الأخطاء أو الحاجة للدعم؟ ولماذا لا تعني الدلالة الإحصائية وحدها نجاحًا تجاريًا؟
10. **الدقيقة العاشرة — القرار:** **{recommendation}**. أكدي أن هذه **محاكاة تعليمية**، وأن الإطلاق الحقيقي يحتاج تجربة واقعية ومراجعة جودة المساعد والخصوصية والتكلفة.
'''
    (ROOT / 'day_04_report_and_presentation/presentation_10_min.md').write_text(presentation, encoding='utf-8')
    print('[اليوم الرابع] ' + recommendation)
    return recommendation


if __name__ == '__main__':
    from day_01_question_and_design.research_design import run as day_one
    from day_02_sample_size_and_duration.power_and_timeline import run as day_two
    from day_03_analysis_plan_and_execution.synthetic_trial import run as day_three
    study = day_one()
    plan = day_two(study)
    dataset, checks, estimates = day_three(study, plan)
    run(study, plan, checks, estimates)
