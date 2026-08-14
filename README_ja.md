# Foam-Agent    <a href="https://arxiv.org/abs/2505.04997"><img src="https://img.shields.io/badge/arXiv-2505.04997-b31b1b.svg" alt="Paper"></a>
<p align="center">
  <img src="overview.png" alt="Foam-Agent System Architecture" width="800">
</p>

<p align="center">
    <em>OpenFOAM による CFD シミュレーション自動化フレームワーク</em>
</p>

<p align="center">
  <a href="README.md">English</a> | <b>日本語</b>
</p>

**Foam-Agent** は、自然言語のプロンプト1つから **OpenFOAM** による CFD シミュレーションのワークフロー全体を自動化します。メッシュ生成、ケースの設定、実行、エラー修正、後処理までを一貫して管理しますので、数値流体力学に必要とされる専門知識の敷居を大きく下げられます。110件のシミュレーション課題からなる [FoamBench](https://arxiv.org/abs/2509.20374) による評価では、Claude Opus 4.6 を用いて **成功率100%** を達成しました。

網羅的な解説と対話形式での質問には [deepwiki.com/csml-rpi/Foam-Agent](https://deepwiki.com/csml-rpi/Foam-Agent) をご利用ください。

## 主な特徴

- **エンドツーエンドの自動化**: メッシュ生成(外部の Gmsh `.msh` ファイルを含みます)から HPC へのジョブ投入、ParaView や PyVista による可視化まで、プロンプト1つで完結します。
- **マルチエージェントのワークフロー**: Architect、Input Writer、Runner、Reviewer の各エージェントが LangGraph のパイプライン上で連携し、エラーを自動的に修正します(最大25回)。
- **RAG による生成の強化**: OpenFOAM のチュートリアルから構築した階層的な FAISS インデックスが文脈に応じた参照を提供しますので、設定ファイルを正確に生成できます。
- **組み替え可能なサービス構成**: 中核の機能を MCP ツールとして公開しますので、Claude Code、Cursor をはじめとするエージェント環境と連携できます。

## クイックスタート

### 1. Docker イメージを取得して実行する

```bash
docker run -it \
  -e OPENAI_API_KEY=your-key-here \
  -p 7860:7860 \
  --name foamagent \
  leoyue123/foamagent
```

このコンテナーには OpenFOAM v10、Python、および依存関係一式が導入済みです。

> 特定のリリースを使う場合: `docker pull leoyue123/foamagent:v2.0.0`

### 2. プロンプトを記述する

コンテナー内の `user_requirement.txt` を編集します。

```text
do a Reynolds-Averaged Simulation (RAS) pitzdaily simulation. Use PIMPLE algorithm.
The domain is a 2D millimeter-scale channel geometry. Boundary conditions specify a
fixed velocity of 10m/s at the inlet (left), zero gradient pressure at the outlet
(right), and no-slip conditions for walls. Use timestep of 0.0001 and output every
0.01. Finaltime is 0.3. use nu value of 1e-5.
```

### 3. 実行する

```bash
uv run python foambench_main.py --output ./output --prompt_path ./user_requirement.txt
```

以上です。Foam-Agent がケースを計画し、OpenFOAM のファイルをすべて生成し、シミュレーションを実行して、エラーを自動的に修正します。

## 設定

設定項目はすべて `src/foamagent/config.py` にあり、妥当な既定値が入っています。各項目は環境変数で上書きできますので、ファイルを編集する必要はありません。Docker や CI で特に有用です。

### LLM のプロバイダーとモデル

| 環境変数 | 用途 | 指定できる値 |
|---|---|---|
| `FOAMAGENT_MODEL_PROVIDER` | LLM のバックエンド(既定は `openai`) | `openai`、`openai-codex`、`anthropic`、`bedrock`、`ollama` |
| `FOAMAGENT_MODEL_VERSION` | モデルの識別子(既定は `gpt-5-mini`) | 例: `gpt-5-mini`、`gpt-5.3-codex`、`claude-opus-4-6` |
| `FOAMAGENT_OPENAI_BASE_URL` | OpenAI 互換エンドポイント(OpenRouter、vLLM、LiteLLM など) | ベース URL。空の場合は OpenAI の公式エンドポイントを使います |

記述例を下記に示します。

```bash
docker run -it \
  -e FOAMAGENT_MODEL_PROVIDER=anthropic \
  -e ANTHROPIC_API_KEY=your-key-here \
  -e FOAMAGENT_MODEL_VERSION=claude-opus-4-6 \
  -p 7860:7860 \
  leoyue123/foamagent
```

### 埋め込みのプロバイダーとモデル

| 環境変数 | 用途 | 指定できる値 |
|---|---|---|
| `FOAMAGENT_EMBEDDING_PROVIDER` | 埋め込みのバックエンド | `openai`、`huggingface`、`ollama` |
| `FOAMAGENT_EMBEDDING_MODEL` | 埋め込みモデル | 例: `Qwen/Qwen3-Embedding-0.6B`、`text-embedding-3-small` |

既定では `huggingface` の `Qwen/Qwen3-Embedding-0.6B` を使います。ローカルで動作しますので、API キーは不要です。

### API キー

| 変数 | 必要となる場面 |
|---|---|
| `OPENAI_API_KEY` | `openai` プロバイダーを使う場合 |
| `ANTHROPIC_API_KEY` | `anthropic` プロバイダーを使う場合 |
| AWS の認証情報 | `bedrock` プロバイダーを使う場合 |

### OpenFOAM の実行方式

既定では、実行中のマシンに導入された OpenFOAM でソルバーを実行します。これをコンテナー内での実行に切り替えられますので、ホストに OpenFOAM がない場合に有用です。

| 環境変数 | 用途 | 既定値 |
|---|---|---|
| `FOAMAGENT_OPENFOAM_RUNTIME` | `native` はローカルの導入を読み込み、`docker` はイメージ内で実行します | `native` |
| `FOAMAGENT_OPENFOAM_IMAGE` | `docker` 方式で使うイメージ | `foam-bench:latest` |
| `FOAMAGENT_OPENFOAM_BASHRC` | そのイメージ内の OpenFOAM の bashrc のパス | `/opt/openfoam10/etc/bashrc` |

`docker` 方式では、ケースディレクトリをコンテナー内の同一の絶対パスへマウントしますので、ログに現れるパスがホスト側と一致します。呼び出し元の UID と GID を渡しますので、生成されたファイルが root 所有になりません。

### その他の設定

| 環境変数 | 用途 | 既定値 |
|---|---|---|
| `FOAMAGENT_MAX_LOOP` | エラー修正の反復回数の上限 | `25` |
| `FOAMAGENT_MAX_TIME_LIMIT` | ソルバーの実行を打ち切るまでの秒数 | `3600` |
| `FOAMAGENT_LOG_LEVEL` | ログの詳細度。ログは標準エラーへ出力し、標準出力には CLI の標識のみを流します | `INFO` |
| `FOAMAGENT_ROOT` | `database/` と `runs/` を探す位置 | リポジトリのルート |

### Input Writer の生成モード

`src/foamagent/config.py` の `input_writer_generation_mode` で指定します。

| モード | 挙動 | 適した用途 |
|---|---|---|
| `sequential_dependency` | ファイルを順に生成し、生成済みのファイルを文脈として使います | 実行が高価な場合(HPC、長時間のシミュレーション) |
| `parallel_no_context` | ファイルを並行して生成し、ファイル間の文脈を使いません | 再試行が安価なローカル実行 |

### 推奨するモデル

| フレームワーク | モデル | Basic | Advanced |
|---|---|---:|---:|
| FoamAgent 2.0.0 (10 loops) | Opus 4.6 | 85.45% | 100% |
| FoamAgent 2.0.0 (25 loops) | Opus 4.6 | 100% | 100% |
| FoamAgent 2.0.0 (25 loops) | Sonnet 4.6 | 87.88% | 75.00% |
| FoamAgent 2.0.0 (25 loops) | Haiku 4.6 | 54.55% | 37.50% |
| FoamAgent 2.0.0 (25 loops) | gpt-5.4 | 45.45% | 75.00% |
| FoamAgent 2.0.0 (25 loops) | gpt-5.3-codex | 54.55% | 62.50% |

最良の結果を得るには **Anthropic Claude Opus 4.6** を推奨します。

## 応用的な使い方

### メッシュファイルの持ち込み

Foam-Agent は外部の Gmsh `.msh` ファイル(ASCII 2.2 形式)に対応しています。境界条件をプロンプトに記述し、メッシュを渡してください。

```bash
uv run python foambench_main.py \
  --output ./output \
  --prompt_path ./user_req_tandem_wing.txt \
  --custom_mesh_path ./tandem_wing.msh
```

ホスト側のメッシュファイルを Docker へマウントする場合は下記のようにします。

```bash
docker run -it \
  -e OPENAI_API_KEY=your-key-here \
  -v /path/to/my_mesh.msh:/home/openfoam/Foam-Agent/my_mesh.msh \
  -p 7860:7860 \
  leoyue123/foamagent
```

### Skill と MCP の連携(Claude Code、Cursor、Windsurf など)

Foam-Agent は CFD のワークフロー全体を **MCP サーバー** として公開します。MCP は Claude Code、Cursor、Windsurf をはじめとする AI ツールが共通して対応するプロトコルです。あわせて、コマンド1つでシミュレーションを実行する **Claude Code の Skill**(`/foam`)を同梱しています。

#### 設定手順(ローカルへ導入する場合)

```bash
# 1. 導入します(foamagent-mcp コマンドが追加されます)
uv sync --extra rag-local --extra direct-api --extra viz

# 2. お使いの AI ツールへ登録します
claude mcp add foamagent -- foamagent-mcp                # Claude Code
```

**Cursor** の場合は、Settings > Features > MCP > Edit MCP Settings を開き、下記を追加します。

```json
{
  "mcpServers": {
    "foamagent": {
      "command": "foamagent-mcp"
    }
  }
}
```

**Windsurf およびその他の MCP 対応ツール**でも、同じ JSON の設定を使えます。

#### 設定手順(Docker の場合)

Docker で実行する場合は、HTTP サーバーを起動して MCP クライアントから接続します。

```bash
docker run -it \
  -e OPENAI_API_KEY=your-key-here \
  -p 7860:7860 \
  leoyue123/foamagent \
  foamagent-mcp --transport http --host 0.0.0.0 --port 7860
```

続いて MCP クライアント側を設定します。

```json
{
  "mcpServers": {
    "foamagent": {
      "url": "http://localhost:7860/mcp"
    }
  }
}
```

> リモートのサーバー上で Docker を実行する場合は、ポート7860へ到達できるようにしてください(SSH のポート転送や `-p 7860:7860` を使います)。

#### 利用できる MCP ツール

Foam-Agent は既定で **Foundation OpenFOAM v10** の規約に従って出力を生成します。`FOAMAGENT_OPENFOAM_FORK=esi` を設定した場合、生成された入力ファイルを ESI OpenFOAM(`openfoam.com`)の命名規約と辞書の規約へ、可能な範囲で変換してから返します。実行、レビュー、修正のワークフローは、現時点では主に Foundation OpenFOAM v10 で検証しています。

| ツール | 説明 |
|------|-------------|
| `plan` | 要件を解析し、Foundation v10 の参照を用いてシミュレーションの構成を計画します |
| `input_writer` | OpenFOAM の設定ファイルを生成します。`FOAMAGENT_OPENFOAM_FORK=esi` の場合は生成したファイルを変換します |
| `run` | Allrun スクリプトをローカルで実行し、エラーを収集します。主に Foundation OpenFOAM v10 で検証しています |
| `review` | シミュレーションのエラーを解析し、Foundation v10 の参照を用いて LLM が修正方針を提示します |
| `apply_fixes` | レビューの結果に基づいて OpenFOAM のファイルを書き換えます。ESI のケースは可能な範囲での対応となります |
| `visualization` | PyVista でシミュレーション結果を可視化します |

#### Claude Code の Skill

このリポジトリをクローンした Claude Code の利用者向けに、`/foam` という Skill を `.claude/skills/foam.md` に同梱しています。MCP のツール群を1つのワークフローとして組み立てます。

```
/foam Simulate lid-driven cavity flow at Re=1000
```

これにより、計画、ファイル生成、実行、レビューと修正の反復、可視化までの一連の処理が動作します。

### Codex の OAuth サインイン(API キーを使わない方法)

ChatGPT や Codex の契約をお持ちの場合、API キーの代わりに OAuth で認証できます。このプロバイダーは明示的に選択したときだけ有効になります。他のツールのトークンキャッシュをディスクから読み出すため、既定では選択されません。

1. ホストマシンに [Codex CLI](https://github.com/openai/codex) を導入します。
2. `codex login` を実行し、**"Sign in with ChatGPT"** を選びます。
3. トークンキャッシュが存在することを確認します: `ls ~/.codex/auth.json`
4. コンテナーへマウントします。

```bash
docker run -it \
  -e FOAMAGENT_MODEL_PROVIDER=openai-codex \
  -e FOAMAGENT_MODEL_VERSION=gpt-5.3-codex \
  -v ~/.codex/auth.json:/root/.codex/auth.json:ro \
  -p 7860:7860 \
  leoyue123/foamagent
```

Foam-Agent は下記の順序で OAuth のトークンを探索し、最初に見つかったものを使います。

- `$CODEX_HOME/auth.json`
- `~/.codex/auth.json`
- `~/.clawdbot/agents/main/agent/auth-profiles.json`

> セキュリティーに関する注意: `auth.json` にはアクセストークンが含まれます。パスワードと同様に扱ってください。

### 手動での導入(Docker を使わない場合)

依存関係は [uv](https://docs.astral.sh/uv/) で管理します。

```bash
git clone https://github.com/csml-rpi/Foam-Agent.git
cd Foam-Agent

# database/ 配下の FAISS インデックスは Git LFS で管理しています。この手順を省くと
# 約130バイトのポインターファイルのままとなり、検索が失敗します。
git lfs install --local && git lfs pull

# コアのみを導入します。必要な追加依存(extras)は下表から選んでください。
uv sync --extra rag-local --extra direct-api --extra viz
```

コアの導入では重い依存を意図的に除いています。使う機能に応じて追加依存を選んでください。

| 追加依存 | 内容 | 必要となる場面 |
|---|---|---|
| `rag-local` | FAISS、sentence-transformers、torch(CPU 版) | 同梱のチュートリアルデータベースを検索する場合(既定の構成) |
| `direct-api` | langchain-openai、langchain-anthropic、openai、anthropic | このプロセス内で推論を実行する場合 |
| `viz` | PyVista | 後処理の画像を生成する場合 |
| `web` | FastAPI、uvicorn | `app.py` のウェブ UI を使う場合 |
| `hpc` | boto3 | SLURM や HPC へ投入する場合 |
| `ollama`、`bedrock` | 各プロバイダーの SDK | 該当するプロバイダーを使う場合 |
| `all` | 上記のすべて | |

あわせて **Foundation OpenFOAM v10**([openfoam.org](https://openfoam.org))の導入と読み込みが必要です。これが既定であり、完全に検証された実行経路です。ESI OpenFOAM(`openfoam.com`)向けのファイル生成は `FOAMAGENT_OPENFOAM_FORK=esi` の設定により可能な範囲で変換しますが、ESI での実行と修正の反復についてはケースごとの確認をお願いします。[Foundation v10 の公式な導入手順](https://openfoam.org/version/10/)に従い、下記で確認してください。

```bash
echo $WM_PROJECT_DIR   # 例えば /opt/openfoam10 と表示されます
```

続いて実行します。

```bash
uv run python foambench_main.py --output ./output --prompt_path ./user_requirement.txt
```

### ソースから Docker イメージを構築する

```bash
git clone https://github.com/csml-rpi/Foam-Agent.git
cd Foam-Agent
docker build -f docker/Dockerfile -t foamagent:latest .
docker run -it \
  -e OPENAI_API_KEY=your-key-here \
  -p 7860:7860 \
  foamagent:latest
```

## 開発

```bash
uv sync                          # コアと開発用ツール
uv run pytest -m "not integration" -q
uv run ruff check .
```

単体テストは、API の認証情報、ネットワーク、Docker、Git LFS の実体、torch のいずれも必要としません。この制約が `import foamagent` に副作用を持たせないための担保となっていますので、新しい単体テストもこの範囲に収めてください。同梱のデータベースを必要とするテストには `integration` の印を付けており、既定では除外されます。

CI は push とプルリクエストのたびに、lint、Python 3.10 と 3.12 での単体テスト、およびホイールの構築を実行します。チェックアウトでは意図的に Git LFS の実体を取得しません。

## トラブルシューティング

| 症状 | 対処 |
|---|---|
| OpenFOAM の環境が見つからない | 意図した OpenFOAM の bashrc が読み込まれているか確認してください。検証済みの既定は Foundation OpenFOAM v10([openfoam.org](https://openfoam.org))です。ESI OpenFOAM を使う場合は `FOAMAGENT_OPENFOAM_FORK=esi` の設定とケースごとの確認が必要です |
| データベースのファイルが見つからない | `database/` を含めてリポジトリ全体をクローンしているか確認してください。Docker イメージには構築済みのものが入っています |
| `Index type 0x73726576 ("vers") not recognized` | FAISS インデックスが Git LFS の未取得のポインターです。`git lfs install --local && git lfs pull` を実行してください |
| 依存関係が足りない | `uv sync --all-extras` を実行してください |
| API キーのエラー | 該当するキー(`OPENAI_API_KEY`、`ANTHROPIC_API_KEY` など)が設定されているか確認してください |
| MCP の接続エラー | コンテナーが動作しており、ポート7860へ到達できるか確認してください |

> **OpenFOAM のバージョンについて:** Foam-Agent は既定で **Foundation OpenFOAM v10**([openfoam.org](https://openfoam.org))を対象とします。`FOAMAGENT_OPENFOAM_FORK=esi` を設定した場合、生成したファイルを ESI OpenFOAM([openfoam.com](https://openfoam.com)、例えば v2312、v2406、v2512)の規約へ可能な範囲で変換します。Docker イメージには Foundation OpenFOAM v10 が導入済みです。

## コミュニティー

### WeChat コミュニティーへの参加

中国語圏の利用者は、ボランティアの WeChat アカウント **ZDSJTUCFD** を追加することで Foam-Agent の WeChat コミュニティーに参加できます。ボランティアがグループへ招待します。

## 引用

Foam-Agent を研究で利用される場合は、下記の論文を引用してください。

```bibtex
@article{yue2025foam,
  title={Foam-Agent: Towards Automated Intelligent CFD Workflows},
  author={Yue, Ling and Somasekharan, Nithin and Zhang, Tingwen and Cao, Yadi and Chen, Zhangze and Di, Shimin and Pan, Shaowu},
  journal={arXiv preprint arXiv:2505.04997},
  year={2025}
}

@article{somasekharan2026cfdllmbench,
    title={CFDLLMBench: A Benchmark Suite for Evaluating Large Language Models in Computational Fluid Dynamics},
    author={Somasekharan, Nithin and Yue, Ling and Cao, Yadi and Li, Weichao and Emami, Patrick and Bhargav, Pochinapeddi Sai and Acharya, Anurag and Xie, Xingyu and Pan, Shaowu},
    journal={Journal of Data-centric Machine Learning Research},
    year={2026},
    url={https://openreview.net/forum?id=kTcH1MnkjY},
    note={}
}

```

## スター数の推移

[![Star History Chart](https://api.star-history.com/svg?repos=csml-rpi/Foam-Agent&type=timeline&legend=top-left)](https://www.star-history.com/#csml-rpi/Foam-Agent&type=timeline&legend=top-left)
