Set WShell = CreateObject("WScript.Shell")
WShell.CurrentDirectory = "c:\Users\Administrator\Downloads\DeepRetail (1)\DeepRetail"
WShell.Run "cmd /c python run_demo.py --reset --anomaly --source 0 --no-display >> logs\pipeline.log 2>&1", 0, False
