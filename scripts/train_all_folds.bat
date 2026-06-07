@echo off
setlocal

REM Train all five classifier folds with the supplied config, or the EfficientNetV2-S default.
set CONFIG=%~1
if "%CONFIG%"=="" set CONFIG=configs\classifier\efficientnetv2_s.yml

for %%F in (1 2 3 4 5) do (
  echo Training fold %%F with %CONFIG%
  python scripts\train_cls.py --config "%CONFIG%" --fold %%F
  if errorlevel 1 exit /b %errorlevel%
)

endlocal
