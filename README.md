# Preference Optimization Reproduction

Repo nhỏ để reproduce và train preference optimization cho bài toán toán học, bắt đầu với DPO trên Hugging Face TRL.

## Mục tiêu

- Reproduce DPO với cấu hình được version-control.
- Chạy smoke test offline trên dữ liệu mẫu.
- Có thể thay model/dataset mà không sửa code.
- Lưu metrics, checkpoint và config cho từng run.

## Cài đặt

Yêu cầu Python 3.10+ và GPU CUDA cho training thực tế.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
pip install -e '.[dev]'
```

## Smoke test không cần tải model

```bash
python -m src.prepare_data --input data/toy_preferences.jsonl --output-dir artifacts/toy
pytest -q
```

## Train DPO

Mặc định dùng base model `Qwen/Qwen2.5-1.5B-Instruct`, reward model `Qwen/Qwen2.5-0.5B-Instruct` và bộ preference toán local. Đổi các model trong `config.py` nếu cần:

```bash
accelerate launch scripts/train_dpo.py --config configs/dpo.yaml
```

Ví dụ thay model và dataset:

```bash
accelerate launch scripts/train_dpo.py \
  --config configs/dpo.yaml \
  --model-name-or-path Qwen/Qwen2.5-0.5B-Instruct \
  --dataset-name HuggingFaceH4/ultrafeedback_binarized
```

Dataset cần có ba cột chuẩn: `prompt`, `chosen`, `rejected`. Nếu dataset có format khác, chuẩn hóa ở `src/prepare_data.py` trước khi train.

### Dùng dữ liệu toán lớn hơn

`open-r1/OpenR1-Math-220k` là dataset lời giải để SFT, không phải preference pairs nên không thể đưa thẳng vào DPO. Với DPO, hãy tạo cặp lời giải đúng/sai bằng math verifier rồi lưu cùng schema `prompt`, `chosen`, `rejected`. Bộ local hiện tại chỉ là smoke dataset để kiểm tra pipeline, không đủ lớn để huấn luyện chất lượng cao.

## Train reward model trên Anthropic HH-RLHF

Reward model dùng `Qwen/Qwen2.5-0.5B-Instruct` mặc định và tối ưu pairwise preference loss:
`-log(sigmoid(score(chosen) - score(rejected)))`. Dataset `Anthropic/hh-rlhf` có sẵn hai cột `chosen` và `rejected`, nên không cần chuẩn hóa thêm.

```bash
pip install -r requirements.txt
```

```bash
python main.py
```

Mọi tham số nằm trong `config.py`. Để chạy smoke test nhanh, đổi `max_train_samples` thành `1000`, `max_eval_samples` thành `200` và `max_test_samples` thành `200`. Script lưu checkpoint tokenizer/model và `metrics.json`, trong đó có `eval.loss` và `eval.accuracy`.

Config mặc định dùng hai GPU (`gpu_ids: [0, 1]`), `batch_size: 2`, gradient accumulation và sequence length 256. Nếu vẫn hết VRAM, giảm `max_length` xuống 128, giảm `batch_size` xuống 1 (khi đó hai GPU không được tận dụng đều), và kiểm tra process cũ bằng `nvidia-smi`.

Chạy training bằng:

```bash
python main.py
```

Model được phân phối qua hai GPU bằng `DataParallel` và checkpoint được lưu ở dạng model bình thường để `experiment.py` load lại được.

Training in progress và metrics từng epoch ra terminal; metrics tổng hợp được lưu tại `outputs/reward_model_hh_rlhf/metrics.json`. Logic train nằm trong `train.py`, còn entry point chạy train nằm trong `main.py`.

## Test-time alignment với Best-of-N

Sau khi train reward model, chạy experiment:

```bash
python experiment.py \
  --reward-model outputs/reward_model_hh_rlhf \
  --max-prompts 100 \
  --output outputs/test_time_alignment/results.json
```

Experiment dùng test split riêng, cùng một pool response được sinh bởi base model và lấy prefix cho `N = 1, 2, 4, 8`. Reward model chấm từng response, sau đó chọn response có reward cao nhất. Kết quả lưu toàn bộ prompt, candidate, reward, response baseline và response được chọn trong `results.json`.

Đây là phần đo reward-model selection, chưa phải đánh giá chất lượng độc lập. Cần dùng human evaluation hoặc một judge model cố định để đo win-rate của baseline và response được chọn; không nên dùng chính reward model làm ground truth.

## Đánh giá

```bash
python scripts/evaluate.py \
  --model-path outputs/dpo \
  --dataset data/toy_preferences.jsonl \
  --output artifacts/eval.json
```

Evaluator hiện đo tỷ lệ model chọn câu trả lời `chosen` theo log-probability; đây là sanity check, không phải đánh giá chất lượng cuối cùng.

## Cấu trúc

```text
configs/        Cấu hình thí nghiệm YAML
data/           Preference pairs mẫu
scripts/        Entry points train/eval
src/            Code chuẩn hóa dữ liệu và tiện ích
 tests/         Smoke tests
```

## Reproducibility checklist

- Ghi lại commit, config, model revision và dataset revision.
- So sánh cùng split và cùng max sequence length.
- Theo dõi `loss`, `rewards/chosen`, `rewards/rejected`, `rewards/margins`.
