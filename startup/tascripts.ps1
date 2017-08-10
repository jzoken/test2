Start-Process -NoNewWindow python  C:\Projects\tireProgramming\testPython\generate_imagesfromftpM.py  > f:/TireAuditRoot/Logs/gi.txt
Start-Process -NoNewWindow python  C:\Projects\tireProgramming\testPython\generate_masksM.py      > f:/TireAuditRoot/Logs/gm.txt
Start-Process -NoNewWindow python  C:\Projects\tireProgramming\testPython\generate_masksMSmart.py      > f:/TireAuditRoot/Logs/gmsm.txt
Start-Process -NoNewWindow python  C:\Projects\tireProgramming\testPython\generate_codesM.py     > f:/TireAuditRoot/Logs/gc.txt
Start-Process -NoNewWindow python  C:\Projects\tireProgramming\testPython\generate_reportsM.py   > f:/TireAuditRoot/Logs/gr.txt
Start-Process -NoNewWindow python  C:\Projects\tireProgramming\testPython\generate_emailResultsM.py  > f:/TireAuditRoot/Logs/gem.txt
Start-Process -NoNewWindow -FilePath C:\"Program Files"\Agisoft\"PhotoScan Pro"\photoscan -ArgumentList '-r C:\Projects\tireProgramming\testPython\generate_3DM.py > f:/TireAuditRoot/Logs/g3d.txt'
Start-Process -NoNewWindow -FilePath C:\"Program Files"\Agisoft\"PhotoScan Pro"\photoscan -ArgumentList '-r C:\Projects\tireProgramming\testPython\generate_3DMSmart.py > f:/TireAuditRoot/Logs/g3dsm.txt'
Start-Process  sqlite3 F:\TireAuditRoot\tireProcessState.db
Start-Process  explorer C:\Projects\ftp\TireScan\Incoming
Start-Process  explorer F:\TireAuditRoot\TireScans
Start-Process "C:\Program Files (x86)\FileZilla Server\FileZilla Server Interface.exe"
cmd /c start powershell -NoExit -Command {ipconfig}


