"""Single entry point: run all four course-day modules, in order."""
from day_01_question_and_design.research_design import run as first_day
from day_02_sample_size_and_duration.power_and_timeline import run as second_day
from day_03_analysis_plan_and_execution.synthetic_trial import run as third_day
from day_04_report_and_presentation.evidence_report import run as fourth_day

def main():
    study = first_day()
    plan = second_day(study)
    dataset, checks, estimates = third_day(study, plan)
    fourth_day(study, plan, checks, estimates)
    print('\n لعرض النتائج تحقق من outputs/results و outputs/charts.')

if __name__ == '__main__':
    main()
