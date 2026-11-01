# 基本操作

コマンドはリポジトリのルートで実行する。

## 収録

```bash
dora run dataflow/episode-record.yaml
```

起動中に http://127.0.0.1:8080 を開く。Start でエピソードが始まり、End で保存され、そのあと Success か Fail を選ぶ。Cancel は捨てる。Success / Fail が付いていないエピソードは、あとの変換で飛ばされる。

保存先は `record/pick-and-place` である。`dataflow/episode-record.yaml` の record ノードが `DIRECTORY: ../record`、`NAME: pick-and-place` で、ノードの作業ディレクトリは `dataflow/` だからである。

## 遠隔推論

policy server のアドレスを、リポジトリ直下の `.env` に `host:port` の形で書く。このファイルは git に入らない。

```bash
SERVER_ADDRESS=<host>:<port>
```

```bash
dora run dataflow/remote_inference.yaml
```

起動中に http://127.0.0.1:8081 を開く。同じ変数をシェルで渡したときは、そちらが使われる。

## 時刻合わせの確認

`tools/curation.py` は数を表示するだけで、ファイルは書かない。基準カメラは overhead。`--slop` はハイフン2つで、基準フレームから他のカメラと leader の指令を拾う窓（秒）である。デフォルトは 0.020 秒。基準フレームの 5%（0.5割）以上が落ちたエピソードは `reject` と出る。閾値は `--max-drop` で、デフォルトは 0.05 である。

```bash
uv run tools/curation.py record/pick-and-place --reference overhead --slop 0.020
```

## LeRobot への変換

同じ `--slop` を渡す。leader の指令が action、follower の実測が `observation.state` になる。

```bash
uv run tools/record2lerobot.py record/pick-and-place --root datasets/so101-pick-and-place --reference overhead --slop 0.020
```

成功だけ入れるときは `--success-only` を足す。基準フレームの 5% 以上が落ちたエピソードは書かず、`conversion.json` にも残さない。同じ `--root` へ再度実行すると、`meta/conversion.json` に載っている収録番号は飛ばされ、fps と slop が違うと拒否される。

## Hub へ送る

Hugging Face にログインした状態で実行する。リポジトリは非公開で、名前はログイン中のユーザーの `so101` になる。

```bash
uv run tools/push_dataset.py datasets/so101-pick-and-place
```

既存のリポジトリを指定するときは `--repo-id <user>/<name>`、公開するときは `--public` を足す。
