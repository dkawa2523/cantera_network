# mechanisms/

このディレクトリにベンチマークで使う反応機構（YAML）を置きます。

## 推奨（Cantera同梱）
- `gri30.yaml`（GRI-Mech 3.0, 325反応）

Cantera の data ディレクトリからコピーできます：

```bash
python benchmarks/scripts/setup_mechanisms.py --dest benchmarks/assets/mechanisms
```

Cantera が import できない場合でも、`CANTERA_DATA` に data ディレクトリを指定しておくとコピーできます：

```bash
export CANTERA_DATA=/path/to/cantera/data
python benchmarks/scripts/setup_mechanisms.py --dest benchmarks/assets/mechanisms
```

もし Cantera の import ができない場合は `--download` で GitHub から取得します：

```bash
python benchmarks/scripts/setup_mechanisms.py --dest benchmarks/assets/mechanisms --download
```

参照（公式）:
- Cantera docs: https://cantera.org/stable/userguide/input-tutorial.html
- Cantera GitHub: https://github.com/Cantera/cantera/blob/main/data/gri30.yaml
