from pathlib import Path
wet = Path(r"C:\Projects\AmpItUp!\machine-learning-amp-it-up-project\data\wet\Ibanez-TSA15-ChCrunch")
print("exists:", wet.exists())
for p in sorted(wet.rglob("*"))[:40]:
    print(p.relative_to(wet))