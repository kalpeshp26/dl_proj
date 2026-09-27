Set WShell = CreateObject("WScript.Shell")
WShell.CurrentDirectory = "c:\Users\Administrator\Downloads\DeepRetail (1)\DeepRetail"
WShell.Run "cmd /c python -m uvicorn src.backend.main:app --host 0.0.0.0 --port 8000 >> logs\backend.log 2>&1", 0, False
