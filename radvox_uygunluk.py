# -*- coding: utf-8 -*-
"""RadVox Uygunluk Aracı — bu bilgisayar RadVox'u nasıl çalıştırır?

Tek dosya, yalnızca standart kütüphane. Böylece PyInstaller ile ~10 MB'lik
tek bir exe olur ve hedef makineye hiçbir şey kurulmadan çalışır.

    python tools/radvox_uygunluk.py            # ölçer, HTML belgeyi açar
    python tools/radvox_uygunluk.py --json     # ham veriyi stdout'a basar
    python tools/radvox_uygunluk.py --konsol   # yalnız metin ozet
    python tools/radvox_uygunluk.py --cikti X.html --acma

Tasarim ilkesi: "NVIDIA yoksa çalışmaz" DEMEZ. Uc ayrı kademe raporlar ve
her kademede hangi özelliklerin çalıştığını açıkça listeler. Dikte, RadVox'un
bir parçası — tamamı değil.

Tek dosyalık exe (hiçbir bağımlılık yok, ~10 MB):

    pyinstaller --onefile --console --name RadVox-Uygunluk ^
                tools/radvox_uygunluk.py

Hız ve WER sayıları ONGORUDUR, bu makinede ölçülmüş degerler değildir;
model büyüklüğü, donanım sınıfı ve saha gözlemine dayali tahmindir. Rapor
bunu her yerde açıkça yazar.
"""

import argparse
import html as _html
import json
import os
import platform
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import webbrowser
from datetime import datetime

ARAC_SURUM = "1.2"
URUN = "RadVox.AI"

# ─────────────────────────────────────────────────────────────────────
# 1) ISTERLER — tek kaynak. Uygulama isterleri degisirse burasi degisir.
# ─────────────────────────────────────────────────────────────────────
ISTER = {
    "ram_gb_min": 8,
    "ram_gb_onerilen": 16,
    "cekirdek_min": 4,
    "cekirdek_onerilen": 8,
    "disk_gb_min": 8,          # paket 4.34 GB zip -> ~5 GB acilmis + kullanici verisi
    "disk_gb_onerilen": 20,
    "vram_gb_large": 6,        # large-v3 fp16 (4.7 GB agirlik + aktivasyon)
    "vram_gb_orta": 4,         # turbo / medium fp16
    "ekran_gen_min": 1366,
    "ekran_gen_onerilen": 1920,
    "win_build_min": 19041,    # Windows 10 2004; altinda PyQt5/PyInstaller sorunlu
    "macos_min": 12,           # Monterey (mlx-whisper)
    "nvidia_surucu_min": 528.0,   # CUDA 12.x calisma zamani
    "cc_min": 6.0,             # CTranslate2 CUDA cekirdek tabani
    "cc_max": 9.0,             # paketteki CUDA 12.1 derlemesinin tavani (sm_90)
    "paket_gb": 5.0,
}

# ─────────────────────────────────────────────────────────────────────
# 2) MODEL TABLOSU
#    hiz  : large-v3 = 1.0 kabul edilerek goreli cozumleme hizi
#    wer  : (ham_alt, ham_ust, duzeltilmis_alt, duzeltilmis_ust) %
#           "duzeltilmis" = RadVox'un 8 adimli son isleme zinciri sonrasi
# ─────────────────────────────────────────────────────────────────────
MODELLER = {
    "large-v3": {
        "ad": "Whisper large-v3",
        "parametre": "1.55 milyar",
        "vram": 4.7, "ram": 3.1, "disk": 3.09, "hiz": 1.0, "hiz_cpu": 1.0,
        "wer": (8, 12, 4, 7),
        "not": "En yüksek doğruluk. Uzun anatomik terimlerde ve nadir "
               "kısaltmalarda farkı burada görürsünüz.",
    },
    "large-v3-turbo": {
        "ad": "Whisper large-v3-turbo",
        "parametre": "809 milyon",
        "vram": 2.6, "ram": 1.7, "disk": 1.62, "hiz": 3.5, "hiz_cpu": 2.5,
        "wer": (9, 14, 5, 8),
        "not": "large-v3'ün 4 katmanlı çözücüsü. Doğruluk farkı küçük, hız "
               "farkı büyük — orta sınıf donanımda en iyi denge.",
    },
    "medium": {
        "ad": "Whisper medium",
        "parametre": "769 milyon",
        "vram": 2.6, "ram": 1.6, "disk": 1.53, "hiz": 2.0, "hiz_cpu": 2.2,
        "wer": (15, 22, 10, 15),
        "not": "Türkçe tıbbi terimlerde belirgin düşüş: 'kostofrenik', "
               "'peribronsiyal' gibi kelimeler bozulmaya başlar.",
    },
    "small": {
        "ad": "Whisper small",
        "parametre": "244 milyon",
        "vram": 1.2, "ram": 0.8, "disk": 0.48, "hiz": 4.0, "hiz_cpu": 4.5,
        "wer": (25, 35, 20, 28),
        "not": "Dikte için önerilmez; düzeltme süresi yazma süresini geçer.",
    },
    "base": {
        "ad": "Whisper base",
        "parametre": "74 milyon",
        "vram": 0.7, "ram": 0.4, "disk": 0.14, "hiz": 8.0, "hiz_cpu": 9.0,
        "wer": (40, 55, 35, 48),
        "not": "Yalnızca 'açılmıyor mu' testi için. Rapor üretilemez.",
    },
}

# Sunum sirasi
MODEL_SIRA = ["large-v3", "large-v3-turbo", "medium", "small", "base"]

# ─────────────────────────────────────────────────────────────────────
# 3) DONANIM SINIF TABLOLARI
#    endeks 1.0 = large-v3 fp16'da ~8x gercek zaman (RTX 3060 sinifi)
# ─────────────────────────────────────────────────────────────────────
GPU_SINIFLARI = [
    # (regex, endeks, aciklama)
    (r"\b(50[89]0|5090|5080)\b", 3.4, "GeForce RTX 50 serisi (Blackwell)"),
    (r"\b(507[05]|5060)\b", 2.0, "GeForce RTX 50 serisi (Blackwell)"),
    (r"\b(4090)\b", 3.0, "GeForce RTX 4090"),
    (r"\b(4080)\b", 2.4, "GeForce RTX 4080"),
    (r"\b(4070)\b", 1.8, "GeForce RTX 4070"),
    (r"\b(4060)\b", 1.1, "GeForce RTX 4060"),
    (r"\b(3090)\b", 2.2, "GeForce RTX 3090"),
    (r"\b(3080)\b", 1.9, "GeForce RTX 3080"),
    (r"\b(3070)\b", 1.4, "GeForce RTX 3070"),
    (r"\b(3060\s*ti)\b", 1.3, "GeForce RTX 3060 Ti"),
    (r"\b(3060|3050)\b", 1.0, "GeForce RTX 30 serisi"),
    (r"\b(2080|2070)\b", 1.0, "GeForce RTX 20 serisi"),
    (r"\b(2060|1660|1650)\b", 0.65, "GeForce RTX 2060 / GTX 16 serisi"),
    (r"\b(10[6-8]0)\b", 0.45, "GeForce GTX 10 serisi (Pascal)"),
    (r"\b(105\d)\b", 0.28, "GeForce GTX 1050 (Pascal, giriş seviyesi)"),
    (r"\b(1030|9[05]0m?|mx\d{3})\b", 0.18, "Giriş seviyesi / dizüstü NVIDIA"),
    (r"\b(h100|h200|b200)\b", 5.0, "Veri merkezi hızlandırıcı"),
    (r"\b(a100|l40|l4|a6000|a5000|a4000)\b", 2.4, "Profesyonel NVIDIA"),
    (r"\b(t4|p100|v100)\b", 0.9, "Veri merkezi (önceki kuşak)"),
    (r"\brtx\s*a\d{4}\b", 1.5, "NVIDIA RTX A serisi"),
    (r"\brtx\s*[3-8]000\b", 2.4, "NVIDIA RTX profesyonel (Ada/Blackwell)"),
    (r"\bquadro\b", 0.8, "Quadro"),
]

APPLE_SINIFLARI = [
    (r"m4\s*(max|ultra)", 1.6, "Apple M4 Max/Ultra"),
    (r"m4\s*pro", 1.35, "Apple M4 Pro"),
    (r"m4", 1.1, "Apple M4"),
    (r"m3\s*(max|ultra)", 1.35, "Apple M3 Max/Ultra"),
    (r"m3\s*pro", 1.1, "Apple M3 Pro"),
    (r"m3", 0.9, "Apple M3"),
    (r"m2\s*(max|ultra)", 1.15, "Apple M2 Max/Ultra"),
    (r"m2\s*pro", 1.0, "Apple M2 Pro"),
    (r"m2", 0.75, "Apple M2"),
    (r"m1\s*(max|ultra)", 0.95, "Apple M1 Max/Ultra"),
    (r"m1\s*pro", 0.85, "Apple M1 Pro"),
    (r"m1", 0.55, "Apple M1"),
]

# CPU kusak carpani — cekirdek basina verim (referans: 2020 sonrasi masaustu)
CPU_KUSAK = [
    # Model numarasi 4 veya 5 haneli olabilir (masaustu 10400, dizustu 1165G7)
    (r"core.{0,6}ultra\s*[3579]", 1.30, "Intel Core Ultra"),
    (r"i[3579][\s-]*1[4-9]\d{2,3}", 1.30, "Intel Core 14. kuşak ve üstü"),
    (r"i[3579][\s-]*1[23]\d{2,3}", 1.20, "Intel Core 12-13. kuşak"),
    (r"i[3579][\s-]*1[01]\d{2,3}", 1.00, "Intel Core 10-11. kuşak"),
    (r"i[3579][\s-]*[89]\d{2,3}", 0.85, "Intel Core 8-9. kuşak"),
    (r"i[3579][\s-]*[4-7]\d{2,3}", 0.65, "Intel Core 4-7. kuşak"),
    (r"i[3579][\s-]*[23]\d{2,3}", 0.45, "Intel Core 2-3. kuşak (çok eski)"),
    (r"ryzen\s*[3579]\s*(pro\s*)?9\d{3}", 1.35, "AMD Ryzen 9000 serisi"),
    (r"ryzen\s*[3579]\s*(pro\s*)?[78]\d{3}", 1.25, "AMD Ryzen 7000/8000 serisi"),
    (r"ryzen\s*[3579]\s*(pro\s*)?[56]\d{3}", 1.05, "AMD Ryzen 5000/6000 serisi"),
    (r"ryzen\s*[3579]\s*(pro\s*)?[34]\d{3}", 0.90, "AMD Ryzen 3000/4000 serisi"),
    (r"ryzen|threadripper", 0.85, "AMD Ryzen"),
    (r"apple\s*m\d", 1.20, "Apple Silicon"),
    (r"xeon", 0.80, "Intel Xeon"),
    (r"celeron|pentium|atom|\bn\d{3,4}\b", 0.35, "Giriş seviyesi CPU"),
    (r"snapdragon|oryon", 0.70, "Qualcomm Snapdragon (Windows on ARM)"),
]

# Tarayici surumu icin kart adindan tipik VRAM tahmini (GB).
# exe bu tabloyu KULLANMAZ — nvidia-smi'den gercek degeri okur. Burada
# yalnizca web surumunun kademe tahmini icin duruyor; anahtarlar
# GPU_SINIFLARI'ndaki aciklama metinleridir.
VRAM_TAHMINI = {
    "GeForce RTX 50 serisi (Blackwell)": 16,
    "GeForce RTX 4090": 24,
    "GeForce RTX 4080": 16,
    "GeForce RTX 4070": 12,
    "GeForce RTX 4060": 8,
    "GeForce RTX 3090": 24,
    "GeForce RTX 3080": 10,
    "GeForce RTX 3070": 8,
    "GeForce RTX 3060 Ti": 8,
    "GeForce RTX 30 serisi": 8,
    "GeForce RTX 20 serisi": 8,
    "GeForce RTX 2060 / GTX 16 serisi": 6,
    "GeForce GTX 10 serisi (Pascal)": 8,
    "GeForce GTX 1050 (Pascal, giriş seviyesi)": 4,
    "Giriş seviyesi / dizüstü NVIDIA": 2,
    "Veri merkezi hızlandırıcı": 80,
    "Profesyonel NVIDIA": 24,
    "Veri merkezi (önceki kuşak)": 16,
    "NVIDIA RTX A serisi": 16,
    "NVIDIA RTX profesyonel (Ada/Blackwell)": 24,
    "Quadro": 5,
    "NVIDIA (sınıflandırılamadı)": 6,
}


# Kademeler
KADEME = {
    "A": ("Tam performans", "#15803d"),
    "B": ("Çalışır — hafif model", "#0d9488"),
    "C": ("Çalışır — dikte CPU'da", "#b45309"),
    "D": ("Elle rapor modu", "#6d28d9"),
    "X": ("Desteklenmiyor", "#b91c1c"),
}


# ─────────────────────────────────────────────────────────────────────
# 4) TESPIT
# ─────────────────────────────────────────────────────────────────────

def _kos(argv, zaman_asimi=25):
    """Alt süreç çalıştır, stdout döndür. Dondurulmuş exe'de konsol açmaz."""
    ek = {}
    if os.name == "nt":
        ek["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)
    try:
        p = subprocess.run(argv, capture_output=True, timeout=zaman_asimi, **ek)
    except Exception:
        return None
    if p.returncode != 0 and not p.stdout:
        return None
    try:
        return p.stdout.decode("utf-8", "replace")
    except Exception:
        return None


def _int(v):
    try:
        return int(float(v))
    except Exception:
        return 0


def _float(v):
    try:
        return float(str(v).strip())
    except Exception:
        return 0.0


def _powershell():
    for exe in ("powershell.exe", "pwsh.exe"):
        yol = shutil.which(exe)
        if yol:
            return yol
    return None


PS_BETIK = r"""
param([string]$Cikti)
$ErrorActionPreference = 'SilentlyContinue'
function G($c) { try { Get-CimInstance $c } catch { $null } }

$os  = G Win32_OperatingSystem | Select-Object -First 1
$cs  = G Win32_ComputerSystem  | Select-Object -First 1
$cpu = G Win32_Processor       | Select-Object -First 1
$gpu = G Win32_VideoController
$snd = G Win32_SoundDevice

$mic = @()
try {
  $mic = @(Get-CimInstance Win32_PnPEntity -Filter "PNPClass='AudioEndpoint'" |
           Where-Object { $_.Name -match 'Mikrofon|Microphone|Line In' } |
           ForEach-Object { $_.Name })
} catch {}

$medya = @()
try { $medya = @(Get-PhysicalDisk | ForEach-Object { [string]$_.MediaType }) } catch {}

$diskler = @()
try {
  $diskler = @(Get-CimInstance Win32_LogicalDisk -Filter "DriveType=3" |
               ForEach-Object { @{ ad=[string]$_.DeviceID; bos=[double]$_.FreeSpace } })
} catch {}

$pil = $false
try { if (G Win32_Battery) { $pil = $true } } catch {}

$o = @{
  os_ad        = [string]$os.Caption
  os_build     = [string]$os.BuildNumber
  os_mimari    = [string]$os.OSArchitecture
  makine       = ("{0} {1}" -f $cs.Manufacturer, $cs.Model)
  cpu_ad       = [string]$cpu.Name
  cpu_cekirdek = [int]$cpu.NumberOfCores
  cpu_mantiksal= [int]$cpu.NumberOfLogicalProcessors
  cpu_mhz      = [int]$cpu.MaxClockSpeed
  ram_bayt     = [double]$cs.TotalPhysicalMemory
  gpu          = @($gpu | ForEach-Object { @{
                     ad = [string]$_.Name
                     surucu = [string]$_.DriverVersion
                     yatay = [int]$_.CurrentHorizontalResolution
                     dikey = [int]$_.CurrentVerticalResolution } })
  ses_aygitlari = @($snd | ForEach-Object { [string]$_.Name })
  mikrofonlar   = $mic
  disk_medya    = $medya
  diskler       = $diskler
  pil           = $pil
}
$json = $o | ConvertTo-Json -Depth 5 -Compress
# Konsol kod sayfasi Turkce karakteri bozuyor; UTF-8 dosyadan okunur.
[System.IO.File]::WriteAllText($Cikti, $json, (New-Object System.Text.UTF8Encoding($false)))
"""


def _medya_adi(medya):
    metin = " ".join(str(m) for m in medya).lower()
    if "ssd" in metin:
        return "SSD"
    if "hdd" in metin:
        return "HDD (mekanik)"
    return ""


def topla_windows_taban(h):
    """PowerShell'siz taban ölçüm — yalnız ctypes ve winreg.

    Hastane makinelerinde PowerShell GPO ile kısıtlı olabilir; o durumda
    -ExecutionPolicy Bypass da işe yaramaz ve CIM sorgusu hiç dönmez.
    Bu fonksiyon her koşulda çalışır, PowerShell başarılı olursa üzerine
    daha iyi verilerle yazılır."""
    import ctypes

    try:                                   # Fiziksel bellek
        class MEMSTAT(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong),
                        ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong),
                        ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong),
                        ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong),
                        ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
        ms = MEMSTAT()
        ms.dwLength = ctypes.sizeof(MEMSTAT)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(ms)):
            h["ram_gb"] = round(ms.ullTotalPhys / 1024**3, 1)
    except Exception:
        pass

    try:                                   # Ekran cozunurlugu (birincil)
        u = ctypes.windll.user32
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except Exception:
            pass
        gen, dik = u.GetSystemMetrics(0), u.GetSystemMetrics(1)
        if gen:
            h["ekran"] = [int(gen), int(dik)]
    except Exception:
        pass

    try:                                   # CPU adi — kayit defterinden
        import winreg
        with winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as k:
            ad = winreg.QueryValueEx(k, "ProcessorNameString")[0]
            if ad:
                h["cpu_ad"] = " ".join(str(ad).split())
    except Exception:
        pass

    try:                                   # Windows yapisi
        import winreg
        with winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"SOFTWARE\Microsoft\Windows NT\CurrentVersion") as k:
            yapi = _int(winreg.QueryValueEx(k, "CurrentBuildNumber")[0])
            if yapi:
                h["os_build"] = yapi
                h["os_ad"] = "Windows %s" % (
                    "11" if yapi >= 22000 else "10")
            try:
                urun = winreg.QueryValueEx(k, "ProductName")[0]
                if urun:
                    # Win11'de bu deger hala "Windows 10 ..." diyebiliyor
                    h["os_ad"] = (str(urun).replace("Windows 10", "Windows 11")
                                  if yapi >= 22000 else str(urun))
            except Exception:
                pass
    except Exception:
        pass


def topla_windows(h):
    topla_windows_taban(h)
    ps = _powershell()
    if not ps:
        return
    bet = tempfile.NamedTemporaryFile("w", suffix=".ps1", delete=False,
                                      encoding="utf-8-sig")
    bet.write(PS_BETIK)
    bet.close()
    cikti = bet.name + ".json"
    try:
        _kos([ps, "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
              "-File", bet.name, "-Cikti", cikti], 60)
        try:
            with open(cikti, encoding="utf-8") as f:
                d = json.load(f)
        except Exception:
            return
    finally:
        for yol in (bet.name, cikti):
            try:
                os.unlink(yol)
            except Exception:
                pass

    h["os_ad"] = d.get("os_ad") or h["os_ad"]
    h["os_build"] = _int(d.get("os_build")) or h["os_build"]
    h["os_mimari"] = d.get("os_mimari") or h["os_mimari"]
    h["makine"] = " ".join((d.get("makine") or "").split())
    if d.get("cpu_ad"):
        h["cpu_ad"] = " ".join(str(d["cpu_ad"]).split())
    h["cekirdek"] = _int(d.get("cpu_cekirdek")) or h["cekirdek"]
    h["is_parcacigi"] = _int(d.get("cpu_mantiksal")) or h["is_parcacigi"]
    h["cpu_mhz"] = _int(d.get("cpu_mhz"))
    if d.get("ram_bayt"):
        h["ram_gb"] = round(float(d["ram_bayt"]) / 1024**3, 1)

    gpular = d.get("gpu") or []
    if isinstance(gpular, dict):
        gpular = [gpular]
    for g in gpular:
        ad = " ".join((g.get("ad") or "").split())
        if not ad:
            continue
        h["ekran_kartlari"].append({"ad": ad, "surucu": g.get("surucu") or ""})
        if g.get("yatay"):
            h["ekran"] = [_int(g["yatay"]), _int(g.get("dikey"))]

    mikler = d.get("mikrofonlar") or []
    if isinstance(mikler, str):
        mikler = [mikler]
    h["mikrofonlar"] = [" ".join(str(m).split()) for m in mikler if m]
    sesler = d.get("ses_aygitlari") or []
    if isinstance(sesler, str):
        sesler = [sesler]
    h["ses_karti_var"] = bool(sesler)
    h["ses_tespiti"] = True
    h["ayrintili_sorgu"] = True

    medya = d.get("disk_medya") or []
    if isinstance(medya, str):
        medya = [medya]
    h["disk_turu"] = _medya_adi(medya)
    h["tasinabilir"] = bool(d.get("pil"))

    diskler = d.get("diskler") or []
    if isinstance(diskler, dict):
        diskler = [diskler]
    en_iyi = 0.0
    for dv in diskler:
        bos = float(dv.get("bos") or 0) / 1024**3
        if bos > en_iyi:
            en_iyi, h["disk_surucu"] = bos, dv.get("ad") or ""
    if en_iyi:
        h["disk_bos_gb"] = round(en_iyi, 1)


def topla_darwin(h):
    def sysctl(anahtar):
        return (_kos(["/usr/sbin/sysctl", "-n", anahtar], 5) or "").strip()

    marka = sysctl("machdep.cpu.brand_string")
    if marka:
        h["cpu_ad"] = marka
    bellek = sysctl("hw.memsize")
    if bellek.isdigit():
        h["ram_gb"] = round(int(bellek) / 1024**3, 1)
    for anahtar in ("hw.perflevel0.physicalcpu", "hw.physicalcpu"):
        v = sysctl(anahtar)
        if v.isdigit():
            h["cekirdek"] = int(v)
            break
    v = sysctl("hw.logicalcpu")
    if v.isdigit():
        h["is_parcacigi"] = int(v)
    surum = (_kos(["/usr/bin/sw_vers", "-productVersion"], 5) or "").strip()
    if surum:
        h["os_ad"] = "macOS " + surum
        h["macos_surum"] = surum
    h["makine"] = sysctl("hw.model")
    h["disk_turu"] = "SSD"
    if h["apple_silicon"] and h["ram_gb"]:
        # Birlesik bellek: GPU ile paylasilir, kabaca %70'i modele ayrilabilir
        h["birlesik_vram_gb"] = round(h["ram_gb"] * 0.7, 1)


def topla_linux(h):
    try:
        with open("/proc/cpuinfo", encoding="utf-8", errors="replace") as f:
            for satir in f:
                if satir.lower().startswith("model name"):
                    h["cpu_ad"] = satir.split(":", 1)[1].strip()
                    break
    except Exception:
        pass
    try:
        with open("/proc/meminfo", encoding="utf-8", errors="replace") as f:
            for satir in f:
                if satir.startswith("MemTotal"):
                    h["ram_gb"] = round(int(satir.split()[1]) / 1024**2, 1)
                    break
    except Exception:
        pass
    h["is_parcacigi"] = os.cpu_count() or 0
    h["cekirdek"] = max(1, h["is_parcacigi"] // 2)


def nvidia_smi(h):
    """Yetkili kaynak: kart adı, VRAM, sürücü ve compute capability."""
    yol = shutil.which("nvidia-smi")
    if not yol and os.name == "nt":
        aday = os.path.join(os.environ.get("ProgramW6432", r"C:\Program Files"),
                            "NVIDIA Corporation", "NVSMI", "nvidia-smi.exe")
        yol = aday if os.path.exists(aday) else None
    if not yol:
        return
    alanlar = "name,memory.total,memory.free,driver_version,compute_cap"
    ham = _kos([yol, "--query-gpu=" + alanlar,
                "--format=csv,noheader,nounits"], 30)
    if not ham:
        # compute_cap eski surucude yok — alani cikarip yeniden dene
        ham = _kos([yol, "--query-gpu=name,memory.total,memory.free,driver_version",
                    "--format=csv,noheader,nounits"], 30)
        if not ham:
            return
    for satir in ham.strip().splitlines():
        parca = [p.strip() for p in satir.split(",")]
        if len(parca) < 4 or not parca[0] or "[" in parca[0]:
            continue
        h["nvidia"].append({
            "ad": parca[0],
            "vram_gb": round(_float(parca[1]) / 1024, 1),
            "vram_bos_gb": round(_float(parca[2]) / 1024, 1),
            "surucu": parca[3],
            "cc": _float(parca[4]) if len(parca) > 4 else 0.0,
        })


def sistemi_topla():
    h = {
        "arac_surum": ARAC_SURUM,
        "zaman": datetime.now().strftime("%d.%m.%Y %H:%M"),
        "bilgisayar": socket.gethostname(),
        "platform": sys.platform,
        "os_ad": platform.platform(),
        "os_build": 0,
        "os_mimari": platform.machine(),
        "macos_surum": "",
        "makine": "",
        "cpu_ad": platform.processor() or platform.machine(),
        "cekirdek": 0,
        "is_parcacigi": os.cpu_count() or 0,
        "cpu_mhz": 0,
        "ram_gb": 0.0,
        "ekran_kartlari": [],
        "nvidia": [],
        "mikrofonlar": [],
        "ses_karti_var": False,
        # Ses aygiti sayimi gercekten kosabildi mi? Kosamadiysa "mikrofon yok"
        # sonucu cikarilamaz — tespit bosluguyla donanim kusuru karistirilmaz.
        "ses_tespiti": False,
        "ayrintili_sorgu": False,
        "disk_bos_gb": 0.0,
        "disk_surucu": "",
        "disk_turu": "",
        "ekran": [0, 0],
        "tasinabilir": False,
        "birlesik_vram_gb": 0.0,
        "apple_silicon": sys.platform == "darwin" and platform.machine() == "arm64",
        "tespit_hatasi": "",
    }
    try:
        if sys.platform == "win32":
            topla_windows(h)
        elif sys.platform == "darwin":
            topla_darwin(h)
        else:
            topla_linux(h)
    except Exception as e:                     # tespit hicbir kosulda cokmesin
        h["tespit_hatasi"] = repr(e)

    if not h["ram_gb"]:
        try:
            h["ram_gb"] = round(os.sysconf("SC_PAGE_SIZE") *
                                os.sysconf("SC_PHYS_PAGES") / 1024**3, 1)
        except Exception:
            pass
    if not h["cekirdek"]:
        h["cekirdek"] = max(1, (os.cpu_count() or 2) // 2)
    if not h["disk_bos_gb"]:
        try:
            h["disk_bos_gb"] = round(
                shutil.disk_usage(os.path.expanduser("~")).free / 1024**3, 1)
        except Exception:
            pass
    try:
        nvidia_smi(h)
    except Exception:
        pass
    return h


# ─────────────────────────────────────────────────────────────────────
# 5) DEGERLENDIRME
# ─────────────────────────────────────────────────────────────────────
# Referans: endeks 1.0 -> large-v3 fp16'da ~8x gercek zaman (RTX 3060 sinifi).
# CPU tarafinda referans 8 modern cekirdek -> large-v3 int8'de ~1x.
GPU_TABAN_XRT = 8.0
CPU_TABAN_XRT = 1.0
# word_timestamps=True (guven filtresi buna bagimli) cozumlemeyi ~%15 yavaslatir
ZAMAN_DAMGASI_CARPANI = 0.85
BASLANGIC_GECIKMESI = {"gpu": 0.4, "cpu": 0.8}


def _esle(tablo, metin):
    m = (metin or "").lower()
    for desen, endeks, ad in tablo:
        if re.search(desen, m):
            return endeks, ad
    return None, ""


def gpu_sinifi(ad):
    endeks, aciklama = _esle(GPU_SINIFLARI, ad)
    if endeks is None:
        return 0.9, "NVIDIA (sınıflandırılamadı)"
    if re.search(r"laptop|mobile|max-q", (ad or "").lower()):
        endeks *= 0.72          # dizustu surumler daha dusuk guc butcesinde
        aciklama += " — dizüstü sürümü"
    return endeks, aciklama


def cpu_endeksi(h):
    carpan, kusak = _esle(CPU_KUSAK, h["cpu_ad"])
    if carpan is None:
        carpan, kusak = 0.85, "Siniflandirilamadi"
    cekirdek = h["cekirdek"] or 4
    # 8 cekirdekten sonrasi cozumlemede dogrusal olcmez
    etkin = min(cekirdek, 8) + max(0, cekirdek - 8) * 0.35
    return round(etkin / 8.0 * carpan, 2), kusak


def cc_tahmin(ad):
    """Sürücü compute_cap vermediyse kart adından tahmin et."""
    m = (ad or "").lower()
    for desen, cc in ((r"\b50\d0\b", 12.0), (r"\b40\d0\b", 8.9),
                      (r"\b30\d0\b", 8.6), (r"\b20\d0\b|\b16\d0\b", 7.5),
                      (r"\b10\d0\b", 6.1), (r"h100|h200", 9.0),
                      (r"a100|a\d000", 8.0), (r"\bt4\b", 7.5)):
        if re.search(desen, m):
            return cc
    return 0.0


def kart_durumu(kart):
    """Bir NVIDIA kartının mevcut RadVox yığınında kullanılabilirliği."""
    cc = kart.get("cc") or cc_tahmin(kart["ad"])
    surucu = _float((kart.get("surucu") or "0").split("-")[0])
    if cc and cc > ISTER["cc_max"]:
        return "yeni", cc, (
            "Kart, paketteki CUDA 12.1 derlemesinden yeni (sm_%d, tavan sm_90). "
            "RadVox çökmez ama GPU'yu kullanamaz, sessizce CPU'ya düşer."
            % int(cc * 10))
    if cc and cc < ISTER["cc_min"]:
        return "eski", cc, (
            "Kart mimarisi (sm_%d) CTranslate2 CUDA derlemesinin altında; "
            "dikte CPU'da çalışır." % round(cc * 10))
    if surucu and surucu < ISTER["nvidia_surucu_min"]:
        return "surucu", cc, (
            "NVIDIA sürücüleri %s — CUDA 12 için en az %.0f gerekir. "
            "Sürücü güncellemesi tek başına sorunu çözer."
            % (kart.get("surucu"), ISTER["nvidia_surucu_min"]))
    return "uygun", cc, ""


def hiz_yaz(x):
    """Hız katsayısı: 10'un altında ondalık, üstünde tam sayı."""
    return ("%.1f" % x) if x < 10 else ("%.0f" % x)


def _sure(saniye_ses, xrt, gecikme):
    if xrt <= 0:
        return None
    return round(gecikme + saniye_ses / xrt, 1)


def model_satirlari(taban_xrt, bellek_gb, bellek_alani, gecikme, secilen,
                    hiz_alani="hiz"):
    satirlar = []
    for ad in MODEL_SIRA:
        m = MODELLER[ad]
        ihtiyac = m[bellek_alani]
        sigar = bellek_gb >= ihtiyac * 1.25 if bellek_gb else False
        xrt = taban_xrt * m[hiz_alani] * ZAMAN_DAMGASI_CARPANI
        satirlar.append({
            "anahtar": ad, "ad": m["ad"], "parametre": m["parametre"],
            "ihtiyac_gb": ihtiyac, "disk_gb": m["disk"], "sigar": sigar,
            "xrt": round(xrt, 1),
            "s15": _sure(15, xrt, gecikme), "s60": _sure(60, xrt, gecikme),
            "s180": _sure(180, xrt, gecikme),
            "wer": m["wer"], "not": m["not"], "secilen": ad == secilen,
        })
    return satirlar


def plan_yap(h):
    p = {"uyarilar": [], "kademe": "C", "dikte": True, "notlar": []}
    ram = h["ram_gb"]

    # ── Blok eden kosullar ────────────────────────────────────────────
    engel = []
    if h["platform"] == "win32" and h["os_build"] and h["os_build"] < ISTER["win_build_min"]:
        engel.append("Windows sürümü çok eski (build %d; en az %d gerekir)."
                     % (h["os_build"], ISTER["win_build_min"]))
    if h["platform"] == "darwin" and h["macos_surum"]:
        try:
            if int(h["macos_surum"].split(".")[0]) < ISTER["macos_min"]:
                engel.append("macOS %s — en az %d (Monterey) gerekir."
                             % (h["macos_surum"], ISTER["macos_min"]))
        except ValueError:
            pass
    if "32" in str(h["os_mimari"]) and "64" not in str(h["os_mimari"]):
        engel.append("32 bit işletim sistemi; RadVox yalnızca 64 bit çalışır.")
    if ram and ram < 4:
        engel.append("Bellek %.1f GB — arayüz bile rahat açılmaz." % ram)
    p["engel"] = engel

    # ── Hangi yol? ────────────────────────────────────────────────────
    en_iyi_kart, en_iyi_durum, en_iyi_cc = None, None, 0.0
    for kart in h["nvidia"]:
        durum, cc, mesaj = kart_durumu(kart)
        kart["durum"], kart["cc_gercek"], kart["mesaj"] = durum, cc, mesaj
        if durum == "uygun" and (en_iyi_durum != "uygun" or
                                 kart["vram_gb"] > (en_iyi_kart or {}).get("vram_gb", 0)):
            en_iyi_kart, en_iyi_durum, en_iyi_cc = kart, durum, cc
        elif en_iyi_kart is None:
            en_iyi_kart, en_iyi_durum, en_iyi_cc = kart, durum, cc
        if mesaj:
            p["uyarilar"].append(("uyari", "%s: %s" % (kart["ad"], mesaj)))

    if not h["nvidia"] and not h["apple_silicon"]:
        yabanci = [g["ad"] for g in h["ekran_kartlari"]
                   if not re.search(r"nvidia|geforce|quadro|tesla",
                                    g["ad"], re.I)]
        if yabanci:
            p["notlar"].append(
                "Ekran kartı (%s) görüntü için yeterli, ancak ses tanımayı "
                "hızlandıramaz: RadVox'un motoru (CTranslate2) yalnızca NVIDIA "
                "CUDA kullanır. AMD/Intel kartı olan makinelerde dikte "
                "CPU'da çalışır." % yabanci[0])

    gpu_kullanilir = en_iyi_durum == "uygun"
    cpu_end, cpu_kusak = cpu_endeksi(h)
    p["cpu_endeks"], p["cpu_kusak"] = cpu_end, cpu_kusak

    if gpu_kullanilir:
        endeks, sinif = gpu_sinifi(en_iyi_kart["ad"])
        # Ekran da ayni kartta; goruntu icin ~1.5 GB ayrilir
        bellek = max(0.0, (en_iyi_kart["vram_gb"] or 0) - 1.5)
        p.update(yol="gpu", motor="faster-whisper + CUDA (float16)",
                 cihaz=en_iyi_kart["ad"], cihaz_sinifi=sinif,
                 compute="float16", bellek_gb=round(bellek, 1),
                 bellek_alani="vram", bellek_etiket="Kullanilabilir VRAM",
                 taban=GPU_TABAN_XRT * endeks, gecikme=BASLANGIC_GECIKMESI["gpu"],
                 endeks=endeks, cc=en_iyi_cc)
    elif h["apple_silicon"]:
        endeks, sinif = _esle(APPLE_SINIFLARI, h["cpu_ad"])
        endeks = endeks or 0.7
        p.update(yol="mlx", motor="mlx-whisper (Apple Metal)",
                 cihaz=h["cpu_ad"], cihaz_sinifi=sinif or "Apple Silicon",
                 compute="float16", bellek_gb=h["birlesik_vram_gb"],
                 bellek_alani="vram", bellek_etiket="Birleşik bellekten ayrılabilir",
                 taban=GPU_TABAN_XRT * endeks,
                 gecikme=BASLANGIC_GECIKMESI["gpu"], endeks=endeks, cc=0.0)
    else:
        p.update(yol="cpu", motor="faster-whisper + CPU (int8)",
                 cihaz=h["cpu_ad"], cihaz_sinifi=cpu_kusak,
                 # RAM okunamadıysa asgari isteri varsay: tespit boşluğu
                 # "hiçbir model sığmaz" sonucuna dönüşmemeli.
                 compute="int8",
                 bellek_gb=max(0.0, (ram or ISTER["ram_gb_min"]) - 3.0),
                 bellek_alani="ram", bellek_etiket="Modele ayrılabilir RAM",
                 taban=CPU_TABAN_XRT * cpu_end,
                 gecikme=BASLANGIC_GECIKMESI["cpu"], endeks=cpu_end, cc=0.0)
    return p


def plan_tamamla(h, p):
    ram = h["ram_gb"]
    sirala = ["large-v3", "large-v3-turbo", "medium", "small"]
    if p["yol"] == "cpu":
        # CPU'da dogruluk degil hiz sinirlayici: gercek zamanin altina
        # dusen model dikte icin kullanilamaz.
        sirala = ["large-v3-turbo", "medium", "small"]

    hiz_alani = "hiz_cpu" if p["yol"] == "cpu" else "hiz"
    p["hiz_alani"] = hiz_alani
    secilen = None
    for ad in sirala:
        m = MODELLER[ad]
        if p["bellek_gb"] < m[p["bellek_alani"]] * 1.25:
            continue
        xrt = p["taban"] * m[hiz_alani] * ZAMAN_DAMGASI_CARPANI
        if p["yol"] == "cpu" and xrt < 0.9 and ad != "small":
            continue
        secilen = ad
        break
    if secilen is None:
        # Sigan en kucugu yine de bul
        for ad in reversed(MODEL_SIRA):
            if p["bellek_gb"] >= MODELLER[ad][p["bellek_alani"]] * 1.25:
                secilen = ad
                break
    p["model"] = secilen
    p["satirlar"] = model_satirlari(p["taban"], p["bellek_gb"],
                                    p["bellek_alani"], p["gecikme"], secilen,
                                    hiz_alani)
    secili_satir = next((s for s in p["satirlar"] if s["secilen"]), None)
    p["secili_satir"] = secili_satir
    p["model_yukleme_sn"] = _model_yukleme(h, p, secilen)

    # ── Dikte pratik mi? ──────────────────────────────────────────────
    xrt = secili_satir["xrt"] if secili_satir else 0.0
    # small/base ile dikte "çalışır" sayılmaz: kendi model notumuz bunu
    # zaten önermiyor (%20+ kelime hatası, düzeltme süresi yazma süresini
    # geçer). Sığıyor olması kullanılabilir olduğu anlamına gelmez.
    p["dikte"] = (bool(secilen) and xrt >= 0.45
                  and secilen not in ("small", "base"))
    if not h.get("ses_tespiti"):
        p["uyarilar"].append(("bilgi", "Ses aygıtları sayılamadı; mikrofon "
                                       "durumu bu raporda doğrulanmadı. "
                                       "Dikte değerlendirmesi mikrofonun "
                                       "takılı olduğu varsayımıyla yapıldı."))
    elif not h["mikrofonlar"] and not h["ses_karti_var"]:
        p["dikte"] = False
        p["uyarilar"].append(("hata", "Ses giriş aygıtı bulunamadı. Dikte için "
                                      "mikrofon şart; USB mikrofon/kulaklık takip "
                                      "aracı yeniden çalıştırın."))
    elif not h["mikrofonlar"]:
        p["uyarilar"].append(("uyari", "Sistemde ses kartı var ama etkin bir "
                                       "mikrofon girişi görünmüyor. Mikrofon "
                                       "takılı değilse normaldir."))

    # ── Kademe ────────────────────────────────────────────────────────
    if p["engel"]:
        p["kademe"] = "X"
        p["dikte"] = False
    elif not p["dikte"]:
        p["kademe"] = "D"
    elif p["yol"] in ("gpu", "mlx"):
        p["kademe"] = "A" if secilen == "large-v3" else "B"
    else:
        p["kademe"] = "C"

    _uyarilari_topla(h, p)
    return p


def _model_yukleme(h, p, secilen):
    if not secilen:
        return None
    gb = MODELLER[secilen]["disk"]
    # Disk okuma + agirlik cozme; SSD'de ~0.55 GB/sn, mekanik diskte ~0.13
    hiz = 0.13 if h["disk_turu"].startswith("HDD") else 0.55
    return round(gb / hiz + (4 if p["yol"] != "cpu" else 3), 0)


def _uyarilari_topla(h, p):
    u = p["uyarilar"]
    ram, disk = h["ram_gb"], h["disk_bos_gb"]
    if h["platform"] == "win32" and not h.get("ayrintili_sorgu"):
        u.append(("bilgi", "Ayrıntılı donanım sorgusu (PowerShell/CIM) bu "
                           "makinede çalışmadı — genellikle kurum güvenlik "
                           "politikası engeller. Rapor doğrudan sistem "
                           "çağrılarıyla üretildi; işletim sistemi, CPU, "
                           "bellek, disk, ekran ve NVIDIA kartı doğrudur. "
                           "Mikrofon listesi ve disk türü eksik."))
    if disk and disk < ISTER["disk_gb_min"]:
        u.append(("hata", "%s sürücüsünde %.0f GB boş yer var; kurulum için "
                          "en az %d GB gerekir (paket ~%.1f GB)."
                  % (h["disk_surucu"] or "Sistem", disk,
                     ISTER["disk_gb_min"], ISTER["paket_gb"])))
    elif disk and disk < ISTER["disk_gb_onerilen"]:
        u.append(("uyari", "Boş disk %.0f GB — kurulum sığar ama hasta "
                           "kayıtları ve günlükler için dar." % disk))
    if ram and ram < ISTER["ram_gb_min"]:
        u.append(("hata", "Bellek %.1f GB. Önerilen %d GB; bu makinede dikte "
                          "yerine elle rapor modu gerçekçi."
                  % (ram, ISTER["ram_gb_onerilen"])))
    elif ram and ram < ISTER["ram_gb_onerilen"]:
        u.append(("uyari", "Bellek %.1f GB — çalışır, ancak PACS ve tarayıcı "
                           "aynı anda açıkken sıkışabilir." % ram))
    gen = (h["ekran"] or [0, 0])[0]
    if gen and gen < ISTER["ekran_gen_min"]:
        u.append(("uyari", "Ekran genişliği %d piksel. RadVox üç panelli "
                           "düzen kullanır; %d piksel altında paneller "
                           "sıkışır." % (gen, ISTER["ekran_gen_min"])))
    if h["disk_turu"].startswith("HDD"):
        u.append(("uyari", "Mekanik disk saptandı. Uygulama çalışır ama model "
                           "ilk yüklemede belirgin yavaşlar; SSD'ye taşımak "
                           "açılışı kısaltır."))
    if h["tasinabilir"] and p["yol"] == "gpu":
        u.append(("bilgi", "Dizüstü bilgisayar: NVIDIA kartı yalnızca prize "
                           "takılıyken ve güç planı 'Yüksek performans' iken "
                           "tam hızda çalışır. Pil modunda dikte süresi iki "
                           "katına çıkabilir."))
    if p["yol"] == "gpu" and p.get("cc") and p["cc"] < 7.0:
        u.append(("uyari", "Kart Pascal kuşağı (sm_%d): float16 hızlandırması "
                           "yok. GPU yine kullanılır ama kazanç sınırlıdır."
                  % round(p["cc"] * 10)))
    if p["yol"] == "cpu" and (p.get("model") or "").startswith("large"):
        u.append(("bilgi", "CPU'da 'large' ile başlayan modeller varsayılan "
                           "ayarda otomatik olarak medium'a düşürülür. Turbo'yu "
                           "CPU'da kullanmak için config.json içinde "
                           "allow_large_cpu = true yapın."))
    if p["yol"] == "cpu" and h["nvidia"]:
        u.append(("bilgi", "NVIDIA kartı var ama bu yığın onu kullanamıyor "
                           "(yukarıdaki kart notuna bakın). Sorun çözülürse "
                           "dikte süresi yaklaşık %dx kısalır."
                  % max(2, round(GPU_TABAN_XRT * gpu_sinifi(h["nvidia"][0]["ad"])[0]
                                 / max(p["taban"], 0.05)))))


def kontrol_listesi(h, p):
    k = []

    def ek(baslik, istenen, bulunan, durum, aciklama=""):
        k.append({"baslik": baslik, "istenen": istenen, "bulunan": bulunan,
                  "durum": durum, "aciklama": aciklama})

    os_durum = "hata" if any("Windows sürümü" in e or "macOS" in e or "32 bit" in e
                             for e in p["engel"]) else "ok"
    ek("İşletim sistemi", "Windows 10 (2004+) / 11 · macOS 12+",
       "%s%s" % (h["os_ad"], " · yapı %d" % h["os_build"] if h["os_build"] else ""),
       os_durum)

    cek = h["cekirdek"]
    ek("CPU", "%d çekirdek (önerilen %d)" % (ISTER["cekirdek_min"],
                                                 ISTER["cekirdek_onerilen"]),
       "%s · %d çekirdek / %d iş parçacığı" % (h["cpu_ad"], cek, h["is_parcacigi"]),
       "ok" if cek >= ISTER["cekirdek_onerilen"] else
       "uyari" if cek >= ISTER["cekirdek_min"] else "hata",
       "Sınıf: %s · göreli hız endeksi %.2f" % (p["cpu_kusak"], p["cpu_endeks"]))

    ram = h["ram_gb"]
    ek("Bellek", "%d GB (önerilen %d GB)" % (ISTER["ram_gb_min"],
                                             ISTER["ram_gb_onerilen"]),
       "%.1f GB" % ram if ram else "okunamadı",
       "bilgi" if not ram else
       "ok" if ram >= ISTER["ram_gb_onerilen"] else
       "uyari" if ram >= ISTER["ram_gb_min"] else "hata")

    disk = h["disk_bos_gb"]
    ek("Boş disk alanı", "%d GB (paket ~%.1f GB)" % (ISTER["disk_gb_min"],
                                                     ISTER["paket_gb"]),
       "%.0f GB boş%s" % (disk, " (%s)" % h["disk_surucu"] if h["disk_surucu"] else "")
       if disk else "okunamadı",
       "bilgi" if not disk else
       "ok" if disk >= ISTER["disk_gb_onerilen"] else
       "uyari" if disk >= ISTER["disk_gb_min"] else "hata",
       "Disk türü: %s" % h["disk_turu"] if h["disk_turu"] else "")

    if h["nvidia"]:
        kart = h["nvidia"][0]
        durum_harita = {"uygun": "ok", "surucu": "uyari",
                        "yeni": "uyari", "eski": "uyari"}
        ek("GPU (dikte hızlandırma)",
           "NVIDIA · %d GB VRAM · sm_60–sm_90 · sürücü %.0f+"
           % (ISTER["vram_gb_large"], ISTER["nvidia_surucu_min"]),
           "%s · %.0f GB VRAM · sürücü %s%s"
           % (kart["ad"], kart["vram_gb"], kart["surucu"],
              " · sm_%d" % round(kart.get("cc_gercek", 0) * 10)
              if kart.get("cc_gercek") else ""),
           durum_harita.get(kart.get("durum"), "uyari"),
           kart.get("mesaj", ""))
    elif h["apple_silicon"]:
        ek("GPU (dikte hızlandırma)", "Apple Silicon (M serisi)",
           "%s · %.1f GB birleşik bellek" % (h["cpu_ad"], h["ram_gb"]), "ok",
           "mlx-whisper Metal üzerinde çalışır; ayrı VRAM gerekmez.")
    else:
        ad = h["ekran_kartlari"][0]["ad"] if h["ekran_kartlari"] else "bulunamadı"
        ek("GPU (dikte hızlandırma)", "NVIDIA (isteğe bağlı)", ad, "bilgi",
           "CUDA yok — dikte CPU'da çalışır. Diğer tüm özellikler etkilenmez.")

    mik = h["mikrofonlar"]
    if not h.get("ses_tespiti"):
        ek("Mikrofon", "Herhangi bir ses girişi (16 kHz)", "sorgulanamadı",
           "bilgi", "Ses aygıtı listesi bu makinede okunamadı; mikrofon "
                    "takılıysa dikte çalışır.")
    else:
        ek("Mikrofon", "Herhangi bir ses girişi (16 kHz)",
           mik[0] if mik else ("ses kartı var, giriş görünmüyor"
                               if h["ses_karti_var"] else "bulunamadı"),
           "ok" if mik else ("uyari" if h["ses_karti_var"] else "hata"),
           "%d giriş aygıtı" % len(mik) if len(mik) > 1 else "")

    gen, dik = (h["ekran"] or [0, 0])[:2]
    ek("Ekran çözünürlüğü", "%d x 768 (önerilen %d x 1080)"
       % (ISTER["ekran_gen_min"], ISTER["ekran_gen_onerilen"]),
       "%d x %d" % (gen, dik) if gen else "okunamadı",
       "ok" if gen >= ISTER["ekran_gen_onerilen"] else
       "uyari" if gen >= ISTER["ekran_gen_min"] else "bilgi")

    ek("İnternet", "Dikte için GEREKMEZ", "Yalnız AI asistan ve güncelleme için",
       "bilgi", "Ses kaydı ve çözümleme tamamen bu bilgisayarda yapılır; "
                "hiçbir hasta verisi dışarı çıkmaz.")
    return k


# Ozellik matrisi — tek kaynak. Web surumu bu listeyi JSON olarak alir,
# boylece metinler iki yerde ayrisamaz. "kosul" sembolik: hem Python hem
# tarayici tarafi ayni adlari degerlendirir.
#   her_zaman -> donanimdan bagimsiz
#   windows   -> Windows'ta tam, macOS'ta farkli yol
#   dikte     -> dikte pratikse
#   gpu       -> dikte GPU/Metal uzerindeyse
#   internet  -> internet ve/veya anahtar gerekir
OZELLIKLER = [
    {"ad": "Rapor editörü ve hızlı rapor giriş formları", "kosul": "her_zaman",
     "not": "Şablondan tam rapor üretimi. Donanımdan bağımsız çalışır."},
    {"ad": "Akıllı tamamlama ve kişisel sözlük", "kosul": "her_zaman",
     "not": "Öğrenen ifade havuzu, kısaltma genişletme; CPU gücü gerektirmez."},
    {"ad": "Ölçüm ve RECIST paneli", "kosul": "her_zaman",
     "not": "Hesaplamalar yerel, anında."},
    {"ad": "Önceki tetkikle karşılaştırma", "kosul": "her_zaman",
     "not": "Lezyon eşleme ve değişim cümleleri; dikte olmadan da kullanılır."},
    {"ad": "Rapor kalite kontrolü", "kosul": "her_zaman",
     "not": "Eksik alan, çelişen ifade ve ölçü tutarsızlığı denetimi."},
    {"ad": "RIS'e biçimli yapıştırma", "kosul": "windows",
     "not": "Windows'ta CF_HTML + RTF ile Word paritesi.",
     "not_alt": "macOS'ta public.html / public.rtf yolu kullanılır."},
    {"ad": "Sesli dikte", "kosul": "dikte",
     "not": "%(cihaz)s üzerinde %(model)s",
     "not_alt": "Bu donanımda dikte pratik değil; program elle yazma "
                "modunda tam işlevli."},
    {"ad": "Sesli noktalama ve komutlar", "kosul": "dikte",
     "not": "'yeni satır', 'nokta', 'paragraf' gibi komutlar dikte ile "
            "birlikte gelir.",
     "not_alt": "Dikte kapalıyken sesli komutlar da devre dışı."},
    {"ad": "GPU hızlandırmalı dikte", "kosul": "gpu",
     "not": "Cümle biter bitmez metin ekranda.",
     "not_alt": "CPU modunda metin birkaç saniye sonra gelir."},
    {"ad": "AI asistan (düzeltme, özet, öneri)", "kosul": "internet",
     "not": "İnternet bağlantısı ve API anahtarı gerekir. Kapatılabilir; "
            "kapalıyken sistem tamamen çevrimdışı çalışır."},
    {"ad": "Otomatik güncelleme", "kosul": "internet",
     "not": "İnternet gerekir."},
]


def ozellik_matrisi(h, p):
    dikte = p["dikte"]
    saglanan = {
        "her_zaman": True,
        "windows": h["platform"] == "win32",
        "dikte": dikte,
        "gpu": p["yol"] in ("gpu", "mlx") and dikte,
        "internet": None,                      # kosullu: karar kullanicinin
    }
    sonuc = []
    for o in OZELLIKLER:
        var = saglanan[o["kosul"]]
        if var is None:
            durum, metin = "kosullu", o["not"]
        elif var:
            durum, metin = "var", o["not"]
        else:
            durum, metin = ("kosullu" if o["kosul"] == "windows" else "yok",
                            o.get("not_alt", o["not"]))
        if "%(cihaz)s" in metin or "%(model)s" in metin:
            metin = metin % {
                "cihaz": p["cihaz"],
                "model": MODELLER[p["model"]]["ad"] if p["model"] else "—"}
        sonuc.append((o["ad"], durum, metin))
    return sonuc


def tablolar():
    """Web sürümünün okuyacağı bilgi tabanı.

    Tarayıcı sürümü VRAM ve sürücü sürümünü göremez, dolayısıyla karar
    mantığı zorunlu olarak daha kaba. Ama model tablosu, GPU/CPU sınıfları,
    WER aralıkları, kademe metinleri ve özellik matrisi TEK kaynaktan —
    buradan — gelir; iki sürümde iki ayrı doğru oluşmaz.
    """
    return {
        "surum": ARAC_SURUM,
        "urun": URUN,
        "ister": ISTER,
        "modeller": MODELLER,
        "model_sira": MODEL_SIRA,
        "gpu_siniflari": [[d, e, a] for d, e, a in GPU_SINIFLARI],
        "apple_siniflari": [[d, e, a] for d, e, a in APPLE_SINIFLARI],
        "cpu_kusak": [[d, e, a] for d, e, a in CPU_KUSAK],
        "vram_tahmini": VRAM_TAHMINI,
        "kademe": {k: list(v) for k, v in KADEME.items()},
        "kademe_metni": {k: list(v) for k, v in KADEME_METNI.items()},
        "ozellikler": OZELLIKLER,
        "sabitler": {
            "gpu_taban_xrt": GPU_TABAN_XRT,
            "cpu_taban_xrt": CPU_TABAN_XRT,
            "zaman_damgasi_carpani": ZAMAN_DAMGASI_CARPANI,
            "baslangic_gecikmesi": BASLANGIC_GECIKMESI,
        },
        "wer_metni": WER_METNI,
    }


def degerlendir(h):
    p = plan_tamamla(h, plan_yap(h))
    return {"donanim": h, "plan": p, "kontroller": kontrol_listesi(h, p),
            "ozellikler": ozellik_matrisi(h, p)}


# ─────────────────────────────────────────────────────────────────────
# 6) HTML BELGE
# ─────────────────────────────────────────────────────────────────────
CSS = """
:root{
  --ink:#111827; --mavi:#3b82f6; --mor:#7C3AED; --gri:#6b7280;
  --cizgi:#e5e7eb; --zemin:#f3f4f6; --kart:#ffffff;
  --ok:#15803d; --uyari:#b45309; --hata:#b91c1c; --bilgi:#1d4ed8;
}
*{box-sizing:border-box}
body{margin:0;background:var(--zemin);color:var(--ink);
  font:15px/1.55 "Segoe UI",system-ui,-apple-system,Arial,sans-serif}
.sayfa{max-width:1080px;margin:0 auto;padding:0 20px 64px}
header{background:var(--kart);border-bottom:3px solid var(--mavi);
  padding:22px 0 18px;margin-bottom:24px}
.ustsatir{max-width:1080px;margin:0 auto;padding:0 20px;
  display:flex;align-items:center;justify-content:space-between;gap:16px;flex-wrap:wrap}
.marka{font:700 30px/1 "Courier New",Courier,monospace;letter-spacing:-.5px}
.marka .r{color:var(--ink)} .marka .v{color:var(--mavi)} .marka .a{color:var(--mor)}
.marka small{display:block;font:600 12px/1.4 "Segoe UI",sans-serif;
  color:var(--gri);letter-spacing:2.4px;margin-top:7px;text-transform:uppercase}
.kimlik{text-align:right;font-size:12.5px;color:var(--gri)}
.kimlik b{display:block;color:var(--ink);font-size:14px}
.karne{background:var(--kart);border-radius:12px;padding:22px 24px;
  box-shadow:0 1px 3px rgba(0,0,0,.09);border-left:8px solid var(--mavi)}
.rozet{display:inline-block;padding:5px 14px;border-radius:999px;
  color:#fff;font-weight:700;font-size:13px;letter-spacing:.4px}
.karne h1{margin:12px 0 6px;font-size:25px;line-height:1.25}
.karne p.ozet{margin:0;color:#374151;font-size:15.5px;max-width:74ch}
.kutular{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));
  gap:14px;margin:22px 0}
.kutu{background:var(--kart);border-radius:10px;padding:15px 17px;
  box-shadow:0 1px 3px rgba(0,0,0,.07);border-top:3px solid var(--mavi)}
.kutu .etiket{font-size:11.5px;text-transform:uppercase;letter-spacing:1.1px;
  color:var(--gri);font-weight:600}
.kutu .deger{font-size:22px;font-weight:700;margin:5px 0 2px;line-height:1.2}
.kutu .alt{font-size:12.5px;color:var(--gri)}
h2{font-size:17px;margin:34px 0 12px;padding-bottom:7px;
  border-bottom:2px solid var(--cizgi)}
h2 span{color:var(--gri);font-weight:400;font-size:13.5px}
table{width:100%;border-collapse:collapse;background:var(--kart);
  border-radius:10px;overflow:hidden;box-shadow:0 1px 3px rgba(0,0,0,.07);font-size:14px}
th{background:#eef2f7;text-align:left;padding:10px 12px;font-size:12.5px;
  text-transform:uppercase;letter-spacing:.5px;color:#374151}
td{padding:10px 12px;border-top:1px solid var(--cizgi);vertical-align:top}
td.sayi{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}
.kaydir{overflow-x:auto;-webkit-overflow-scrolling:touch}
.kaydir table{min-width:760px}
tr.secili td{background:#eff6ff;font-weight:600}
tr.solgun td{color:#9ca3af}
.d{font-weight:700;white-space:nowrap}
.d-ok{color:var(--ok)} .d-uyari{color:var(--uyari)}
.d-hata{color:var(--hata)} .d-bilgi{color:var(--bilgi)}
.aciklama{display:block;color:var(--gri);font-size:12.5px;margin-top:3px;
  font-weight:400;max-width:70ch}
ul.mesaj{list-style:none;padding:0;margin:0}
ul.mesaj li{background:var(--kart);border-left:5px solid var(--gri);
  border-radius:7px;padding:11px 15px;margin-bottom:9px;font-size:14px;
  box-shadow:0 1px 2px rgba(0,0,0,.06)}
li.m-hata{border-left-color:var(--hata)} li.m-uyari{border-left-color:var(--uyari)}
li.m-bilgi{border-left-color:var(--bilgi)}
li b{display:block;font-size:12px;text-transform:uppercase;letter-spacing:.7px;
  margin-bottom:3px}
li.m-hata b{color:var(--hata)} li.m-uyari b{color:var(--uyari)}
li.m-bilgi b{color:var(--bilgi)}
.oz{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:10px}
.oz-sat{background:var(--kart);border-radius:8px;padding:11px 14px;
  box-shadow:0 1px 2px rgba(0,0,0,.06);display:flex;gap:11px;align-items:flex-start}
.im{font-size:17px;line-height:1.35;flex:0 0 auto}
.oz-sat .ad{font-weight:600;font-size:14px}
.oz-sat .nt{font-size:12.5px;color:var(--gri);margin-top:2px}
.yok .ad{color:#9ca3af}
.notlar{background:#fffbeb;border:1px solid #fcd34d;border-radius:9px;
  padding:14px 17px;font-size:13.5px;color:#78350f;margin-top:14px}
.notlar b{display:block;margin-bottom:4px}
details{background:var(--kart);border-radius:9px;margin-top:26px;
  box-shadow:0 1px 2px rgba(0,0,0,.06)}
summary{cursor:pointer;padding:12px 16px;font-size:13.5px;color:var(--gri)}
pre{margin:0;padding:0 16px 16px;font:11.5px/1.5 Consolas,monospace;
  white-space:pre-wrap;word-break:break-word;color:#374151;max-height:340px;overflow:auto}
footer{margin-top:34px;padding-top:16px;border-top:1px solid var(--cizgi);
  font-size:12px;color:var(--gri);line-height:1.7}
.yazdir{position:fixed;right:20px;bottom:20px;background:var(--mor);color:#fff;
  border:0;border-radius:999px;padding:12px 22px;font-size:14px;font-weight:600;
  cursor:pointer;box-shadow:0 4px 14px rgba(124,58,237,.35)}
@media print{
  body{background:#fff} .yazdir{display:none}
  .karne,.kutu,table,.oz-sat,ul.mesaj li{box-shadow:none;border:1px solid var(--cizgi)}
  h2{page-break-after:avoid} table{page-break-inside:avoid;font-size:11.5px}
  .kaydir{overflow:visible} .kaydir table{min-width:0}
  td,th{padding:6px 7px}
  details{display:none}
}
"""


def E(x):
    return _html.escape(str(x if x is not None else ""), quote=True)


KADEME_METNI = {
    "A": ("Bu bilgisayar {urun} uygulamasını tam performansta çalıştırır.",
          "Ekran kartı dikteyi üstlenir; en büyük ve en doğru model olan "
          "{model} sorunsuz yüklenir. Cümleyi bitirdiğinizde metin pratik "
          "olarak aynı anda ekranda olur."),
    "B": ("{urun} bu bilgisayarda tam sürümüyle çalışır.",
          "Ekran kartı dikteyi üstlenir, ancak belleği en büyük modele "
          "yetmediği için {model} kullanılır. Hız farkı hissedilmez; "
          "doğruluk farkı nadir terimlerde ortaya çıkar."),
    "C": ("{urun} çalışır — dikte CPU üzerinde.",
          "Uygun bir NVIDIA kartı bulunmadığı için çözümleme CPU'da "
          "yapılır ve {model} kullanılır. Program eksiksizdir; yalnızca "
          "konuştuktan sonra metnin gelmesi birkaç saniye sürer."),
    "D": ("Sesli dikte için bu bilgisayar yeterli değil.",
          "Üzgünüz — bu donanımda kullanılabilir doğrulukta bir model "
          "çalışmıyor, dikteyi önermiyoruz. Ama {urun} elle rapor yazmak "
          "için hâlâ çok iyi: hızlı rapor giriş formları, akıllı "
          "tamamlama, kişisel sözlük, önceki tetkikle karşılaştırma ve "
          "RIS'e biçimli yapıştırma donanımdan tamamen bağımsız çalışır "
          "ve rapor sürenizi tek başına belirgin kısaltır."),
    "X": ("Bu bilgisayar desteklenen sınırın altında.",
          "Aşağıdaki engeller giderilmeden kurulum önerilmez."),
}


def _kutular(h, p, ozellikler):
    k = []
    if p["kademe"] == "X":
        return [("Durum", "%d engel" % len(p["engel"]),
                 "Kuruluma başlanmadan giderilmeli"),
                ("İşletim sistemi", h["os_ad"][:28] or "—",
                 "İstenen: Windows 10 (2004+) / 11 · macOS 12+"),
                ("Bellek", "%.1f GB" % h["ram_gb"] if h["ram_gb"] else "—",
                 "En az %d GB" % ISTER["ram_gb_min"]),
                ("Boş disk", "%.0f GB" % h["disk_bos_gb"] if h["disk_bos_gb"] else "—",
                 "En az %d GB" % ISTER["disk_gb_min"])]
    k.append(("Motor ve cihaz", p["motor"].split(" +")[0],
              "%s · %s" % (p["cihaz"], p["compute"])))
    s = p["secili_satir"]
    # Dikte önerilmiyorsa model önermek kendi kendiyle çelişir: "small
    # sığıyor" ile "small ile dikte olmaz" aynı belgede duramaz.
    if p["dikte"] and p["model"]:
        m = MODELLER[p["model"]]
        k.append(("Önerilen model", m["ad"].replace("Whisper ", ""),
                  "%s parametre · %.1f GB %s"
                  % (m["parametre"], m[p["bellek_alani"]],
                     p["bellek_alani"].upper())))
    else:
        k.append(("Sesli dikte", "Önerilmiyor",
                  "Kullanılabilir doğrulukta model çalışmıyor"))
    if p["dikte"] and s:
        k.append(("15 saniyelik dikte", "%.1f sn" % s["s15"],
                  "gerçek zamanın %s katı · 1 dakikalık dikte %.1f sn"
                  % (hiz_yaz(s["xrt"]), s["s60"])))
        k.append(("Beklenen kelime hatası", "%%%d–%d" % (s["wer"][2], s["wer"][3]),
                  "RadVox düzeltmeleri sonrası · ham çıktı %%%d–%d"
                  % (s["wer"][0], s["wer"][1])))
    else:
        calisan = sum(1 for _, d, _ in ozellikler if d == "var")
        k.append(("Elle rapor modu", "Tam işlevli",
                  "Şablon, sözlük, karşılaştırma, RIS'e yapıştırma"))
        k.append(("Etkilenmeyen özellik", "%d / %d" % (calisan, len(ozellikler)),
                  "Rapor üretimi donanımdan bağımsız çalışır"))
    return k


def _mesajlar(p):
    if not p["uyarilar"] and not p["engel"]:
        return ""
    basliklar = {"hata": "Engel", "uyari": "Dikkat", "bilgi": "Not"}
    parcalar = ['<h2>Bulgular</h2><ul class="mesaj">']
    for e in p["engel"]:
        parcalar.append('<li class="m-hata"><b>Engel</b>%s</li>' % E(e))
    sira = {"hata": 0, "uyari": 1, "bilgi": 2}
    for seviye, metin in sorted(p["uyarilar"], key=lambda x: sira.get(x[0], 3)):
        parcalar.append('<li class="m-%s"><b>%s</b>%s</li>'
                        % (seviye, basliklar.get(seviye, "Not"), E(metin)))
    for n in p["notlar"]:
        parcalar.append('<li class="m-bilgi"><b>Not</b>%s</li>' % E(n))
    parcalar.append("</ul>")
    return "".join(parcalar)


def _kontrol_tablosu(kontroller):
    satirlar = []
    for k in kontroller:
        simge = {"ok": "✓", "uyari": "!", "hata": "✕", "bilgi": "i"}[k["durum"]]
        acik = ('<span class="aciklama">%s</span>' % E(k["aciklama"])
                if k["aciklama"] else "")
        satirlar.append(
            "<tr><td><b>%s</b>%s</td><td>%s</td><td>%s</td>"
            '<td class="d d-%s">%s</td></tr>'
            % (E(k["baslik"]), acik, E(k["istenen"]), E(k["bulunan"]),
               k["durum"], simge))
    return ("<h2>Donanım kontrolü <span>— istenen ve bulunan</span></h2>"
            "<table><tr><th>Bileşen</th><th>RadVox isteri</th>"
            "<th>Bu bilgisayar</th><th>Durum</th></tr>%s</table>"
            % "".join(satirlar))


def _model_tablosu(p):
    if not p["satirlar"]:
        return ""
    birim = "VRAM" if p["bellek_alani"] == "vram" else "RAM"
    satirlar = []
    for s in p["satirlar"]:
        sinif = ("secili" if (s["secilen"] and p["dikte"])
                 else ("" if s["sigar"] else "solgun"))
        if not s["sigar"]:
            durum, sureler = '<span class="d d-hata">sığmaz</span>', ("—", "—", "—")
        else:
            hizli = s["xrt"] >= 1.0
            durum = ('<span class="d d-%s">%s</span>'
                     % ("ok" if hizli else "uyari",
                        "çalışır" if hizli else "yavaş"))
            sureler = ("%.1f sn" % s["s15"], "%.1f sn" % s["s60"],
                       "%.0f sn" % s["s180"])
        isaretle = s["secilen"] and p["dikte"]
        etiket = E(s["ad"]) + (" &nbsp;<b>← önerilen</b>" if isaretle else "")
        satirlar.append(
            '<tr class="%s"><td>%s<span class="aciklama">%s · disk %.2f GB · %s</span></td>'
            '<td class="sayi">%.1f GB</td><td>%s</td><td class="sayi">%sx</td>'
            '<td class="sayi">%s</td><td class="sayi">%s</td><td class="sayi">%s</td>'
            '<td class="sayi">%%%d–%d</td><td class="sayi"><b>%%%d–%d</b></td></tr>'
            % (sinif, etiket, E(s["parametre"]), s["disk_gb"], E(s["not"]),
               s["ihtiyac_gb"], durum, hiz_yaz(s["xrt"]),
               sureler[0], sureler[1], sureler[2],
               s["wer"][0], s["wer"][1], s["wer"][2], s["wer"][3]))
    return (
        "<h2>Bu bilgisayar hangi modeli çalıştırır? "
        "<span>— süreler öngörüdür, ölçüm değildir</span></h2>"
        '<div class="kaydir"><table><tr><th>Model</th><th>%s</th>'
        "<th>Durum</th><th>Hız</th>"
        "<th>15 sn</th><th>1 dk</th><th>3 dk</th><th>Ham WER</th>"
        "<th>Düzeltilmiş</th></tr>%s</table></div>"
        "<p style='font-size:12.5px;color:#6b7280;margin:8px 2px 0'>"
        "<b>Hız</b>: çözümlemenin gerçek zamanın kaç katı olduğu — 10x, "
        "1 dakikalık konuşmanın 6 saniyede yazıya dökülmesi demektir. "
        "<b>15 sn / 1 dk / 3 dk</b>: o uzunlukta bir dikteden sonra metnin "
        "ekranda belirmesi için beklenen süre (model bir kez yüklendikten "
        "sonra; ilk yükleme ~%s sn). Süreler kelime zaman damgası açıkken "
        "hesaplandı (güven filtresi bunu gerektirir, çözümlemeyi ~%%15 "
        "yavaşlatır).</p>"
        % (birim, "".join(satirlar),
           "%.0f" % p["model_yukleme_sn"] if p["model_yukleme_sn"] else "?"))


WER_METNI = """
<div class="notlar"><b>Bu sayılar öngörüdür, bu bilgisayarda yapılmış ölçüm değildir.</b>
Model büyüklüğü, donanım sınıfı ve saha gözlemine dayanır. Gerçek sonuç
mikrofon kalitesine, ortam gürültüsüne, konuşma hızına ve tetkik türüne
göre değişir — en çok da mikrofona. Kendi sesinizle gerçek rakamı ölçmek
on dakikalık iştir: bilinen bir raporu okuyun, çıktı ile karşılaştırın.
Yaka veya masa mikrofonu yerine gürültü bastırmalı bir kulaklık mikrofonu,
tek başına modeli bir basamak yükseltmeye denk gelir.</div>
"""


def _ozellik_bolumu(ozellikler):
    simge = {"var": ("✓", ""), "kosullu": ("◐", ""), "yok": ("✕", "yok")}
    renk = {"var": "d-ok", "kosullu": "d-uyari", "yok": "d-hata"}
    parcalar = ['<h2>Bu bilgisayarda hangi özellikler çalışır?</h2><div class="oz">']
    for ad, durum, nt in ozellikler:
        im, sinif = simge[durum]
        parcalar.append(
            '<div class="oz-sat %s"><div class="im %s">%s</div>'
            '<div><div class="ad">%s</div><div class="nt">%s</div></div></div>'
            % (sinif, renk[durum], im, E(ad), E(nt)))
    parcalar.append("</div>")
    return "".join(parcalar)


def oneriler(h, p):
    o = []
    if p["engel"]:
        o.append("Önce engelleri giderin: " + " ".join(p["engel"]))
    if p["yol"] == "cpu" and not h["nvidia"]:
        o.append("Dikteyi hızlandırmak isterseniz tek gerekli şey bir NVIDIA "
                 "kartı: 8 GB VRAM'li bir RTX 4060/5060 sınıfı kart bile "
                 "en büyük modeli tam hızda çalıştırır. Kart olmadan da "
                 "program eksiksiz kullanılır.")
    if p["yol"] == "cpu" and h["nvidia"]:
        o.append("Karttaki engel giderilirse (sürücü güncellemesi ya da "
                 "GPU-GECISI.md'deki yığın güncellemesi) aynı makine "
                 "kademe atlar; yeni donanım gerekmez.")
    if p["yol"] == "gpu" and p["model"] != "large-v3":
        o.append("Daha fazla VRAM'li bir kart en büyük modeli açar; ancak "
                 "kazanç doğrulukta küçük bir paydır, hızda değil.")
    if h["ram_gb"] and h["ram_gb"] < ISTER["ram_gb_onerilen"]:
        o.append("Belleği %d GB'a çıkarmak, PACS ve tarayıcı açıkken "
                 "oluşan takılmaları giderir." % ISTER["ram_gb_onerilen"])
    if h["disk_turu"].startswith("HDD"):
        o.append("Kurulumu SSD'ye almak açılış süresini birkaç kat kısaltır; "
                 "dikte hızını etkilemez.")
    if not h["mikrofonlar"]:
        o.append("Mikrofon takip aracı yeniden çalıştırın — dikte dışındaki "
                 "her şey mikrofonsuz da çalışır.")
    o.append("Kurulumu Program Files altına yapmayın. Program çalışırken "
             "kendi klasörüne yazıyor; yetki kısıtlı klasörde bazı özellikler "
             "sessizce çalışmaz.")
    return o


def html_uret(sonuc):
    h, p = sonuc["donanim"], sonuc["plan"]
    kademe_ad, kademe_renk = KADEME[p["kademe"]]
    baslik, ozet = KADEME_METNI[p["kademe"]]
    model_ad = MODELLER[p["model"]]["ad"] if p["model"] else "—"
    bicim = {"urun": URUN, "model": model_ad}
    baslik, ozet = baslik.format(**bicim), ozet.format(**bicim)

    kutular = "".join(
        '<div class="kutu" style="border-top-color:%s"><div class="etiket">%s</div>'
        '<div class="deger">%s</div><div class="alt">%s</div></div>'
        % (kademe_renk, E(a), E(b), E(c))
        for a, b, c in _kutular(h, p, sonuc["ozellikler"]))

    oneri_html = "".join("<li>%s</li>" % E(x) for x in oneriler(h, p))
    ham = json.dumps(sonuc["donanim"], ensure_ascii=False, indent=1)

    return """<!DOCTYPE html>
<html lang="tr"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>%(urun)s Uygunluk Belgesi — %(bilgisayar)s</title>
<style>%(css)s</style></head><body>
<header><div class="ustsatir">
  <div class="marka"><span class="r">Rad</span><span class="v">Vox</span><span class="a">.AI</span>
    <small>Uygunluk Belgesi</small></div>
  <div class="kimlik"><b>%(bilgisayar)s</b>%(makine)s<br>%(zaman)s ·
    araç sürümü %(arac)s</div>
</div></header>
<div class="sayfa">
  <div class="karne" style="border-left-color:%(renk)s">
    <span class="rozet" style="background:%(renk)s">KADEME %(kademe)s · %(kademe_ad)s</span>
    <h1>%(baslik)s</h1>
    <p class="ozet">%(ozet)s</p>
  </div>
  <div class="kutular">%(kutular)s</div>
  %(mesajlar)s
  %(kontroller)s
  %(modeller)s
  %(wer)s
  %(ozellikler)s
  <h2>Öneriler</h2>
  <ul class="mesaj">%(oneriler)s</ul>
  <details><summary>Teknik döküm (destek için) — tespit edilen ham veri</summary>
    <pre>%(ham)s</pre></details>
  <footer>
    Bu belge %(urun)s uygunluk aracının (sürüm %(arac)s) bu bilgisayarda
    yaptığı otomatik ölçümden üretildi. Donanım bilgileri işletim sisteminden
    ve NVIDIA sürücülerinden okundu. <b>Hız ve doğruluk sayıları öngörüdür</b>;
    model büyüklüğü ile donanım sınıfına dayanan tahmindir, bu makinede
    çalıştırılmış bir ölçüm değildir. Belge dışarı veri göndermez.
  </footer>
</div>
<button class="yazdir" onclick="window.print()">Belgeyi yazdır / PDF</button>
</body></html>""" % {
        "urun": E(URUN), "css": CSS, "renk": kademe_renk,
        "bilgisayar": E(h["bilgisayar"]),
        "makine": E(h["makine"]) or E(h["os_ad"]),
        "zaman": E(h["zaman"]), "arac": E(ARAC_SURUM),
        "kademe": p["kademe"], "kademe_ad": E(kademe_ad),
        "baslik": E(baslik), "ozet": E(ozet), "kutular": kutular,
        "mesajlar": _mesajlar(p),
        "kontroller": _kontrol_tablosu(sonuc["kontroller"]),
        "modeller": _model_tablosu(p) if p["model"] and p["kademe"] != "X" else "",
        "wer": WER_METNI if p["dikte"] else "",
        "ozellikler": _ozellik_bolumu(sonuc["ozellikler"]),
        "oneriler": oneri_html, "ham": E(ham),
    }


# ─────────────────────────────────────────────────────────────────────
# 7) KONSOL VE GIRIS
# ─────────────────────────────────────────────────────────────────────

def konsol_ozet(sonuc):
    h, p = sonuc["donanim"], sonuc["plan"]
    kademe_ad = KADEME[p["kademe"]][0]
    cizgi = "-" * 66
    sat = [cizgi, "  %s - UYGUNLUK ÖZETİ   (%s)" % (URUN, h["zaman"]), cizgi,
           "  Bilgisayar : %s  %s" % (h["bilgisayar"], h["makine"]),
           "  Sistem     : %s" % h["os_ad"],
           "  CPU        : %s (%d çekirdek)" % (h["cpu_ad"], h["cekirdek"]),
           "  Bellek     : %.1f GB   Boş disk: %.0f GB %s"
           % (h["ram_gb"], h["disk_bos_gb"], h["disk_turu"])]
    if h["nvidia"]:
        for kart in h["nvidia"]:
            sat.append("  Ekran kartı: %s (%.0f GB, sürücü %s) -> %s"
                       % (kart["ad"], kart["vram_gb"], kart["surucu"],
                          kart.get("durum", "?")))
    else:
        sat.append("  Ekran kartı: NVIDIA yok")
    sat += [cizgi, "  SONUÇ : Kademe %s - %s" % (p["kademe"], kademe_ad),
            "  Motor : %s" % p["motor"]]
    if p["dikte"] and p["secili_satir"]:
        s = p["secili_satir"]
        sat += ["  Model : %s" % MODELLER[p["model"]]["ad"],
                "  Hız   : gerçek zamanın %s katı  (15 sn dikte -> %.1f sn)"
                % (hiz_yaz(s["xrt"]), s["s15"]),
                "  WER   : ham %%%d-%d, düzeltme sonrası %%%d-%d (öngörü)"
                % s["wer"]]
    else:
        sat.append("  Dikte : bu donanımda kapalı — elle rapor modu tam işlevli")
    sat.append(cizgi)
    for seviye, metin in p["uyarilar"]:
        etiket = {"hata": "ENGEL", "uyari": "DİKKAT",
                  "bilgi": "NOT"}.get(seviye, seviye.upper())
        sat.append("  [%s] %s" % (etiket, metin))
    for e in p["engel"]:
        sat.append("  [ENGEL] %s" % e)
    return "\n".join(sat)


def _konsolu_hazirla():
    """Konsolu UTF-8'e al. Windows'ta varsayılan kod sayfası (cp857/cp437)
    Türkçe harfleri basamaz; ş ve ğ soru işaretine döner."""
    if os.name == "nt":
        try:
            import ctypes
            ctypes.windll.kernel32.SetConsoleOutputCP(65001)
        except Exception:
            pass
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def _yaz(metin):
    try:
        print(metin)
    except UnicodeEncodeError:                 # kod sayfası hâlâ dar
        kod = sys.stdout.encoding or "ascii"
        print(metin.encode(kod, "replace").decode(kod))


def main(argv=None):
    _konsolu_hazirla()
    ap = argparse.ArgumentParser(
        description="%s uygunluk aracı — bu bilgisayar RadVox'u nasıl "
                    "çalıştırır?" % URUN)
    ap.add_argument("--cikti", metavar="DOSYA",
                    help="HTML belgenin yazılacağı yol")
    ap.add_argument("--acma", action="store_true",
                    help="HTML'i tarayıcıda açma")
    ap.add_argument("--konsol", action="store_true",
                    help="Yalnız metin özet yaz, HTML üretme")
    ap.add_argument("--json", action="store_true",
                    help="Tüm sonucu JSON olarak stdout'a yaz")
    ap.add_argument("--tablolar", action="store_true",
                    help="Bilgi tabanını JSON olarak yaz (web sürümü için)")
    a = ap.parse_args(argv)

    if a.tablolar:
        _yaz(json.dumps(tablolar(), ensure_ascii=False, indent=1))
        return 0

    sonuc = degerlendir(sistemi_topla())

    if a.json:
        _yaz(json.dumps(sonuc, ensure_ascii=False, indent=1, default=str))
        return 0
    if a.konsol:
        _yaz(konsol_ozet(sonuc))
        return 0

    yol = a.cikti
    if not yol:
        ad = "RadVox-Uygunluk-%s-%s.html" % (
            re.sub(r"[^A-Za-z0-9_-]", "", sonuc["donanim"]["bilgisayar"])[:20]
            or "rapor", datetime.now().strftime("%Y%m%d-%H%M"))
        taban = os.path.dirname(os.path.abspath(sys.argv[0])) \
            if getattr(sys, "frozen", False) else tempfile.gettempdir()
        yol = os.path.join(taban, ad)
    try:
        with open(yol, "w", encoding="utf-8") as f:
            f.write(html_uret(sonuc))
    except OSError:
        yol = os.path.join(tempfile.gettempdir(), os.path.basename(yol))
        with open(yol, "w", encoding="utf-8") as f:
            f.write(html_uret(sonuc))

    _yaz(konsol_ozet(sonuc))
    _yaz("\n  Belge: %s" % yol)
    if not a.acma:
        try:
            webbrowser.open("file:///" + yol.replace("\\", "/"))
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
