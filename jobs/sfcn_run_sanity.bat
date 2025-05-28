@echo off
:: Activate virtual environment
call venv\Scripts\activate.bat

:: Run sanity check training scripts
echo Running SFCN sanity checks...

@REM echo Training SFCN 30-40 age range...
@REM python -m brain_age_pred.scripts.train brain_age_pred/configs/sanity/sfcn_snellius_30_40.yaml

@REM echo Training SFCN 60-70 age range...
@REM python -m brain_age_pred.scripts.train brain_age_pred/configs/sanity/sfcn_snellius_60_70.yaml

echo Training SFCN T2...
python -m brain_age_pred.scripts.train brain_age_pred/configs/sanity/sfcn_snellius_t2.yaml

echo Training SFCN T1 30-40 age range...
python -m brain_age_pred.scripts.train brain_age_pred/configs/sanity/sfcn_snellius_t1_30_40.yaml

echo All sanity checks completed!

:: Deactivate the virtual environment
call deactivate

pause
