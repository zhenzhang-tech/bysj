@echo off
setlocal enabledelayedexpansion
cd /d D:\bysj

echo ================================================
echo   Push graduation project to GitHub
echo   Target: https://github.com/zhenzhang-tech/bysj
echo ================================================
echo.

REM ---- 1. check git exists ----
where git >nul 2>&1
if errorlevel 1 (
  echo [ERROR] git not found. Install Git for Windows first.
  pause
  exit /b 1
)

REM ---- 2. init repo if needed ----
if not exist .git (
  echo [1/7] Initializing git repo...
  git init
) else (
  echo [1/7] .git already exists, skip init
)

REM ---- 3. ensure git identity is set ----
git config user.email >nul 2>&1
if errorlevel 1 (
  echo [2/7] Setting default git identity...
  git config user.name "zhenzhang-tech"
  git config user.email "zhenzhang-tech@users.noreply.github.com"
) else (
  echo [2/7] git identity already configured
)

REM ---- 4. stage everything (respects .gitignore) ----
echo [3/7] Staging files (.gitignore already excludes big files)...
git add .
if errorlevel 1 (
  echo [ERROR] git add failed
  pause
  exit /b 1
)

REM ---- 5. guard: reject any staged file over 100MB (GitHub hard limit) ----
echo [4/7] Checking for files over 100MB...
set BIGFOUND=0
for /f "delims=" %%f in ('git ls-files') do (
  if %%~zf GTR 104857600 (
    echo   [WARNING] over 100MB: %%f  (%%~zf bytes^)
    set BIGFOUND=1
  )
)
if !BIGFOUND!==1 (
  echo.
  echo [ERROR] Files over 100MB found. GitHub will reject the push.
  echo Add them to .gitignore and run this script again.
  pause
  exit /b 1
)
echo   No file over 100MB. OK.

REM ---- 6. commit ----
echo [5/7] Committing...
git commit -m "Graduation project: entity matching API_test"
if errorlevel 1 (
  echo   (nothing to commit, or commit failed - see message above)
)

REM ---- 7. configure remote ----
echo [6/7] Setting remote origin...
git remote get-url origin >nul 2>&1
if errorlevel 1 (
  git remote add origin https://github.com/zhenzhang-tech/bysj
) else (
  git remote set-url origin https://github.com/zhenzhang-tech/bysj
)
git branch -M main

REM ---- 8. push ----
echo [7/7] Pushing to GitHub (first time asks for login: username + Token)...
git push -u origin main

if errorlevel 1 (
  echo.
  echo [FAILED] Push failed. Common causes:
  echo   1. Repo already has files (e.g. README was auto-created):
  echo      run:  git pull origin main --allow-unrelated-histories
  echo      then: git push -u origin main
  echo   2. Network / proxy issue
  echo   3. Token has no repo permission
) else (
  echo.
  echo [SUCCESS] Pushed to https://github.com/zhenzhang-tech/bysj
)

echo.
pause
