"""Day 3 — New e-service-specific simulator, audit and effect estimation.

No example functions from the workshop are copied here. Statistical packages
provide standard published tests and confidence intervals.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import chisquare
from matplotlib import font_manager

try:
    import arabic_reshaper
    from bidi.algorithm import get_display
except ImportError:
    arabic_reshaper = None
    get_display = None


def arabic_label(value):
    """Shape right-to-left chart text when optional rendering packages are available."""
    if arabic_reshaper is None:
        return value
    return get_display(arabic_reshaper.reshape(value))


plt.rcParams['font.family'] = 'DejaVu Sans'

from statsmodels.stats.proportion import proportions_ztest, confint_proportions_2indep
import statsmodels.api as sm
from day_01_question_and_design.research_design import Study

ROOT = Path(__file__).resolve().parents[1]


def group_from_account(account):
    """Deterministic independent group assignment (no per-visit reshuffling)."""
    key = hashlib.sha256(('ai-guide-research-2026|' + account).encode()).digest()
    return 'B' if int.from_bytes(key[:8], 'big') < (1 << 63) else 'A'


def build_data(study, plan):
    rng = np.random.default_rng(study.random_seed)
    total = plan['expected_assigned_total']
    people = np.array([f'account_{k:06d}' for k in range(total)])
    arm = np.array([group_from_account(user) for user in people])
    ai_available = (arm == 'B').astype(int)
    day = np.repeat(np.arange(1, plan['scheduled_days_full_weeks'] + 1), study.new_users_daily)

    # One digital journey per unique account. Latent difficulty affects success
    # but cannot confound treatment because it is generated independently.
    difficulty = rng.beta(2.2, 3.1, size=total)
    previous_visits = rng.poisson(1.3 + 4.0*(1-difficulty))
    baseline_prob = study.baseline_rate + .12*(0.50-difficulty) + .014*(previous_visits-3)
    # Scenario includes a short-lived novelty boost; not an observed business fact.
    novelty = 0.009 * np.exp(-(day-1)/2.5)
    success_prob = np.clip(baseline_prob + ai_available*(study.hypothetical_ai_effect + novelty), .02, .97)
    completed = rng.binomial(1, success_prob)

    error_prob = np.clip(.032 + .008*difficulty + ai_available*study.hypothetical_error_effect, 0, 1)
    support_prob = np.clip(.107 + .018*difficulty + ai_available*study.hypothetical_support_effect, 0, 1)
    error_flag = rng.binomial(1, error_prob)
    contacted_support = rng.binomial(1, support_prob)
    opened_assistant = rng.binomial(1, .60, total) * ai_available

    # Time is only observed for completers; never treat the comparison as causal.
    elapsed = np.maximum(2, rng.normal(27 + 7*difficulty - 2.2*ai_available, 5.5, size=total))
    minutes_to_finish = np.where(completed == 1, elapsed.round(1), np.nan)
    return pd.DataFrame({
        'account_key': people, 'assigned_arm': arm, 'entry_day': day,
        'visits_before_trial': previous_visits, 'finished_in_24h': completed,
        'encountered_error': error_flag, 'needed_human_support': contacted_support,
        'opened_ai_widget': opened_assistant, 'minutes_to_finish': minutes_to_finish
    })


def audit(df, study):
    required = ['account_key','assigned_arm','entry_day','visits_before_trial',
                'finished_in_24h','encountered_error','needed_human_support','opened_ai_widget']
    missing = {col:int(df[col].isna().sum()) for col in required if df[col].isna().sum()}
    duplicates = int(df['account_key'].duplicated().sum())
    invalid_binary = {col:int((~df[col].isin([0,1])).sum()) for col in
                      ['finished_in_24h','encountered_error','needed_human_support','opened_ai_widget']}
    arm_counts = df['assigned_arm'].value_counts()
    counts = [int(arm_counts.get('A',0)),int(arm_counts.get('B',0))]
    if sum(counts) != len(df):
        raise ValueError('تم العثور على مجموعة تجربة غير معروفة')
    chi2, p_srm = chisquare(counts, f_exp=[len(df)*(1-study.allocation_to_ai),len(df)*study.allocation_to_ai])
    inconsistent = int(((df.assigned_arm=='A') & (df.opened_ai_widget==1)).sum())
    suspicious_times = int(((df.finished_in_24h==0) & df.minutes_to_finish.notna()).sum() +
                           ((df.finished_in_24h==1) & df.minutes_to_finish.isna()).sum())
    status = not missing and not duplicates and not any(invalid_binary.values()) and not inconsistent and not suspicious_times and p_srm>=.001
    checks = {'users':len(df), 'A':counts[0], 'B':counts[1],
              'srm_chi_square':float(chi2), 'srm_p_value':float(p_srm),
              'srm_threshold':.001, 'duplicate_accounts':duplicates,
              'missing_required':missing, 'invalid_binary':invalid_binary,
              'ai_used_in_control':inconsistent,
              'impossible_completion_time_records':suspicious_times,
              'structural_missing_time_for_noncompleters':int(df.minutes_to_finish.isna().sum()),
              'data_usable_for_inference':bool(status)}
    if not status:
        raise ValueError('فشلت فحوص جودة البيانات؛ لا يمكن تفسير النتائج السببية: '+str(checks))
    return checks


def compare_binary(df, column):
    counts = df.groupby('assigned_arm')[column].agg(['sum','count'])
    count_a, count_b = int(counts.loc['A','sum']),int(counts.loc['B','sum'])
    n_a, n_b = int(counts.loc['A','count']),int(counts.loc['B','count'])
    a, b = count_a/n_a, count_b/n_b
    # Newcombe interval with Wilson scores, works better than a simplistic Wald CI.
    low, high = confint_proportions_2indep(count_b,n_b,count_a,n_a,compare='diff',method='newcomb',alpha=.05)
    z,p = proportions_ztest([count_b,count_a],[n_b,n_a],alternative='two-sided')
    return {'control_rate':float(a),'ai_rate':float(b),'difference':float(b-a),
            'ci95_lower':float(low),'ci95_upper':float(high),
            'p_two_sided':float(p), 'count_A':n_a,'count_B':n_b}


def adjusted_check(df):
    """Pre-treatment adjustment (ANCOVA), independent from trainer CUPED code."""
    x = pd.DataFrame({'intercept':np.ones(len(df)),
                      'ai_offer':(df.assigned_arm=='B').astype(int),
                      'prior_visits_centered':df.visits_before_trial-df.visits_before_trial.mean()})
    fit = sm.OLS(df.finished_in_24h.astype(float),x).fit(cov_type='HC3')
    interval = fit.conf_int().loc['ai_offer']
    return {'method':'تحليل ANCOVA مع أخطاء معيارية متينة HC3 وضبط الزيارات السابقة للمعالجة',
            'difference':float(fit.params['ai_offer']), 'ci95_lower':float(interval.iloc[0]),
            'ci95_upper':float(interval.iloc[1]), 'p_two_sided':float(fit.pvalues['ai_offer'])}


def make_figures(df, effects):
    out = ROOT/'outputs/charts';out.mkdir(parents=True,exist_ok=True)
    first = effects['primary']
    fig,ax=plt.subplots(figsize=(7,4))
    bars=ax.bar([arabic_label('الواجهة الحالية (A)'), arabic_label('إتاحة المساعد (B)')],
                [100*first['control_rate'],100*first['ai_rate']],color=['#397b91','#e5a05d'])
    ax.bar_label(bars,fmt='%.2f%%',padding=3)
    ax.set_ylim(0,85);ax.set_ylabel(arabic_label('نسبة الإكمال خلال 24 ساعة (%)'));ax.set_title(arabic_label('الإكمال لدى جميع المستخدمين الموزعين عشوائيًا'))
    fig.tight_layout();fig.savefig(out/'01_completion.png',dpi=150);plt.close(fig)

    data = [(arabic_label('الأثر الأساسي ITT'), effects['primary']), (arabic_label('تحليل الحساسية ANCOVA'), effects['ancova'])]
    fig,ax=plt.subplots(figsize=(7.5,3.8))
    for y,(name,res) in enumerate(data):
        ax.errorbar([res['difference']*100],[y],xerr=[[100*(res['difference']-res['ci95_lower'])],
                    [100*(res['ci95_upper']-res['difference'])]],fmt='o',capsize=4)
    ax.axvline(0,color='gray',linestyle='--')
    ax.axvline(5,color='gray',linestyle=':',label=arabic_label('الحد العملي: 5 نقاط مئوية'))
    ax.set_yticks(range(len(data)),[v[0] for v in data]);ax.set_xlabel(arabic_label('الأثر بالنقاط المئوية (فترة ثقة 95%)'))
    ax.legend();ax.set_title(arabic_label('حجم الأثر وعدم اليقين'))
    fig.tight_layout();fig.savefig(out/'02_effect_intervals.png',dpi=150);plt.close(fig)

    trends = df.groupby(['entry_day','assigned_arm']).finished_in_24h.mean().unstack()
    fig,ax=plt.subplots(figsize=(8,4))
    ax.plot(trends.index,100*(trends['B']-trends['A']),marker='o',label=arabic_label('الفرق اليومي'))
    ax.axhline(first['difference']*100,linestyle='--',color='#e5a05d',label=arabic_label('الفرق الإجمالي'))
    ax.axhline(0,color='gray',linewidth=.8)
    ax.set(xlabel=arabic_label('يوم الدخول'),ylabel=arabic_label('الفرق بالنقاط المئوية'),title=arabic_label('تغير الأثر عبر أيام التجربة (استكشافي)'))
    ax.legend();fig.tight_layout();fig.savefig(out/'03_daily_pattern.png',dpi=150);plt.close(fig)

    names,values=[],[]
    for name,k in [(arabic_label('الأخطاء'),'errors'),(arabic_label('التواصل مع الدعم'),'support')]:
        res=effects[k];names.append(name);values.append([res['control_rate']*100,res['ai_rate']*100])
    fig,ax=plt.subplots(figsize=(7,4))
    x=np.arange(2);w=.34
    ax.bar(x-w/2,[v[0] for v in values],w,label='A',color='#397b91')
    ax.bar(x+w/2,[v[1] for v in values],w,label='B',color='#e5a05d')
    ax.set_xticks(x,names);ax.set_ylabel(arabic_label('النسبة (%)'));ax.legend();ax.set_title(arabic_label('مقاييس الحماية'))
    fig.tight_layout();fig.savefig(out/'04_guardrails.png',dpi=150);plt.close(fig)


def run(study=None, plan=None):
    study=study or Study()
    if plan is None:
        from day_02_sample_size_and_duration.power_and_timeline import run as power_run
        plan=power_run(study)
    df=build_data(study,plan)
    checks=audit(df,study)
    analyses={
        'primary':compare_binary(df,'finished_in_24h'),
        'ancova':adjusted_check(df),
        'errors':compare_binary(df,'encountered_error'),
        'support':compare_binary(df,'needed_human_support'),
        'ai_widget_opened_among_B':float(df.loc[df.assigned_arm=='B','opened_ai_widget'].mean()),
        'minutes_among_completers_only':{
            v:float(df.loc[(df.assigned_arm==v)&(df.finished_in_24h==1),'minutes_to_finish'].mean()) for v in ['A','B']},
        'time_note':'الزمن مقاس فقط لمن أكملوا الخدمة بعد المعالجة؛ المقارنة وصفية وليست تقديرًا سببيًا.'
    }
    output=ROOT/'outputs/results';output.mkdir(parents=True,exist_ok=True)
    (ROOT/'data').mkdir(exist_ok=True)
    df.to_csv(ROOT/'data/synthetic_service_journeys.csv',index=False)
    (output/'day_03_integrity.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
    (output/'day_03_estimates.json').write_text(json.dumps(analyses,indent=2),encoding='utf-8')
    make_figures(df,analyses)
    print(f"[اليوم الثالث] قيمة p لفحص SRM: {checks['srm_p_value']:.4f}؛ الإكمال A: {analyses['primary']['control_rate']:.2%}، B: {analyses['primary']['ai_rate']:.2%}")
    return df, checks, analyses

if __name__ == '__main__':
    run()
