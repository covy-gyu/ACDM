# -*- mode: python ; coding: utf-8 -*-


block_cipher = None


a = Analysis(
    ['run.py'],
    pathex=[],
    binaries=[],
    datas=[

        ('jamo/*.py','jamo'),
        ('jamo/data','jamo/data'),
    
        ('infer/Laser','infer/Laser'),

        ('data/model/*.pt','data/model'),

        ('infer/POCR/*.py','infer/POCR'),
        ('infer/POCR/pororo/*.py','infer/POCR/pororo'),
        ('infer/POCR/pororo/tasks/*.py','infer/POCR/pororo/tasks'),
        ('infer/POCR/pororo/tasks/utils/*.py','infer/POCR/pororo/tasks/utils'),
        ('infer/POCR/pororo/models/brainOCR/*.py','infer/POCR/pororo/models/brainOCR'),
        ('infer/POCR/pororo/models/brainOCR/modules/*.py','infer/POCR/pororo/models/brainOCR/modules'),

        ('infer/Powder/*.py','infer/Powder'),
        
        ('infer/Wing/*.py','infer/Wing'),
        
        ('infer/YOLOS/*.py','infer/YOLOS'),

        ('logs/*.py','logs'),
        
        ('ultralytics/hub/*.py','ultralytics/hub'),
        ('ultralytics/nn/*.py','ultralytics/nn'),
        ('ultralytics/tracker/cfg','ultralytics/tracker/cfg'),
        ('ultralytics/tracker/trackers/*.py','ultralytics/tracker/trackers'),
        ('ultralytics/tracker/utils/*.py','ultralytics/tracker/utils'),
        ('ultralytics/tracker/*.py','ultralytics/tracker'),
        ('ultralytics/yolo/cfg/*.py','ultralytics/yolo/cfg'),
        ('ultralytics/yolo/cfg/*.yaml','ultralytics/yolo/cfg'),
        ('ultralytics/yolo/data/dataloaders/*.py','ultralytics/yolo/data/dataloaders'),
        ('ultralytics/yolo/data/*.py','ultralytics/yolo/data'),
        ('ultralytics/yolo/engine/*.py','ultralytics/yolo/engine'),
        ('ultralytics/yolo/utils/callbacks/*.py','ultralytics/yolo/utils/callbacks'),
        ('ultralytics/yolo/utils/*.py','ultralytics/yolo/utils'),
        ('ultralytics/yolo/v8/classify/*.py','ultralytics/yolo/v8/classify'),
        ('ultralytics/yolo/v8/detect/*.py','ultralytics/yolo/v8/detect'),
        ('ultralytics/yolo/v8/segment/*.py','ultralytics/yolo/v8/segment'),
        ('ultralytics/yolo/v8/*.py','ultralytics/yolo/v8'),
        ('ultralytics/yolo/*.py','ultralytics/yolo'),
        ('ultralytics/*.py','ultralytics'),
    ],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='ACDM v2.5',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    icon='icon.png',
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
