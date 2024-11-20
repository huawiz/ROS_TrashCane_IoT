# 獲取當前目錄下所有的jpg檔案
$files = Get-ChildItem -Path . -Filter *.jpg

# 初始化計數器
$counter = 50

# 計算需要的補零位數
$padLength = [Math]::Floor([Math]::Log10($files.Count)) + 1

# 為每個檔案重新命名
foreach ($file in $files | Sort-Object Name) {
    # 建立新檔名 (例如: 01.jpg, 02.jpg, etc.)
    $newName = "{0}.jpg" -f $counter.ToString("D$padLength")
    
    # 檢查新檔名是否已存在
    if (Test-Path $newName) {
        # 如果檔案已存在，建立臨時檔名
        $tempName = "temp_" + $newName
        Rename-Item -Path $file.Name -NewName $tempName
        Write-Host "暫時重命名: $($file.Name) -> $tempName"
    } else {
        # 直接重新命名
        Rename-Item -Path $file.Name -NewName $newName
        Write-Host "重新命名: $($file.Name) -> $newName"
    }
    
    $counter++
}

# 處理所有臨時檔名
$tempFiles = Get-ChildItem -Path . -Filter temp_*.jpg
foreach ($tempFile in $tempFiles) {
    $finalName = $tempFile.Name.Replace("temp_", "")
    Rename-Item -Path $tempFile.Name -NewName $finalName
    Write-Host "完成重命名: $($tempFile.Name) -> $finalName"
}

Write-Host "`n重新命名完成！共處理 $($files.Count) 個檔案。"