# 自動打軸機 · Auto Subtitle

面向直播剪輯的本機字幕工具，支援日文、中文、英語與韓文。匯入影片或音檔，透過 faster-whisper 與 Silero VAD 生成字幕，匯出 ASS／SRT 後交由 Aegisub 人工修整。預設忠實保留辨識文字，不翻譯或自動潤稿。

目前版本 0.9.9，開發重點為 Windows 10／11；Windows 10 實機相容性仍待確認，macOS 暫緩。模型、媒體與字幕專案保留在本機，模型首次使用時需要下載。

## 安裝與執行

需要 Python 3.11–3.13。Windows PowerShell：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e '.[dev]'
.\.venv\Scripts\python.exe -m autosub
```

FFmpeg 預設由 imageio-ffmpeg 提供，也可用 `AUTOSUB_FFMPEG` 指定。`AUTOSUB_MODEL_DIR` 可指定模型快取位置；模型選單亦支援本機 CTranslate2 模型資料夾。

## 使用

1. 匯入媒體並選擇音軌，以 1–20 分鐘單檔為主要情境。
2. 選語言與模型。CPU 可從 `small` 開始；品質比較可選 `large-v3` 或 `turbo`，`tiny` 適合流程測試。
3. CPU 使用 INT8；相容的 NVIDIA GPU 可選 CUDA／FP16。遊戲名與人名可填入詞彙提示，但不保證正確辨識。
4. 開始辨識並指定專案位置。取消會保留完整辨識批次，之後可接續；進度會保存在指定的專案位置。
5. 選取字幕後按「試聽選取句」，按需調整生成參數並匯出 ASS／SRT，最後在 Aegisub 校正。

「詞彙提示」可選擇本機儲存的清單；按「管理詞彙」可建立清單、新增單詞、修改或刪除。每行一個單詞，儲存後可重複使用；取消不寫入。原專案仍保存原有提示文字，修改或刪除清單不會回寫已存專案。

介面預設採用淺色配色；主畫面隱藏生成提示欄與篩選。字幕表預設唯讀，內建修軸工具隱藏。F5 暫停／原位置續播，Ctrl+↑／↓ 切換字幕。

新日文專案保留 Whisper 原生句界，過長時才細分；舊專案沿用原策略。「套用參數」使用已有詞級資料重新分段，不重跑模型。句首先行、尾留、最短顯示時間及短空隙延長規則見 [打軸規格](docs/timing-rules.md)。ASS／SRT 預設不自動插入兩行換行，可在進階設定啟用。

「專案與工具」預設隱藏；需要既有專案工具時可在啟動前設定 `AUTOSUB_ENABLE_PROJECT_TOOLS=1`。

專案存檔遇 Windows 拒絕替換／共享錯誤時，最多重試五次，總等待約 0.75 秒。持續失敗仍會回報錯誤，舊檔與待存的 `.autosub.json.tmp` 保留；不會更改權限或刪除舊檔。此版本的辨識與分段策略不變。

專有名詞、輕聲、多人重疊及 BGM 仍可能造成錯字、漏句或句界問題。詞彙提示僅輔助辨識，不強制替換；本工具不保證聲學邊界精度或完整度。

請保留媒體、`*.autosub.json`、分析快取與外置模型。媒體搬移可重新連結；舊版專案升級存檔前會保存備份，新版專案勿用舊程式覆寫。新辨識的解碼診斷會與進度一同保存在本機專案中。

## Windows 打包

```powershell
.\.venv\Scripts\python.exe scripts/build.py --output-name AutoSubtitle-0.9.9
```

GPU 版需先安裝額外執行庫，再依序建置：

```powershell
.\.venv\Scripts\python.exe -m pip install -e '.[gpu]'
.\.venv\Scripts\python.exe scripts/build.py --with-cuda --output-name AutoSubtitle-GPU-0.9.9
```

模型不隨 exe 打包。GPU 版會收集 CUDA／cuDNN 執行庫與授權聲明，仍需相容的 NVIDIA 驅動。保留整個可攜目錄，包括 `_internal` 及 GPU 版的 `cuda`。根目錄 `Start-AutoSubtitle.cmd` 可啟動本機已建置版本。詳見 [Windows 操作說明](docs/windows-quickstart.txt)。

## 開發與驗證

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe scripts/check_repository.py
```

測試以合成音訊、臨時檔案及模擬模型檢查程式行為，不代表真實直播辨識率。通用離線診斷工具位於 `scripts/`；需要的研究計畫與輸入資料須由呼叫者另行提供。

倉庫只保存程式、一般化測試、依賴與公開使用文件。私人素材、字幕成品、辨識輸出、研究計畫及階段報告不提交，詳見 [貢獻與資料規則](CONTRIBUTING.md)。

「試聽選取句」只播放選句的開始至結束時間，不額外延伸前後範圍；循環與暫停續播沿用同一範圍。
