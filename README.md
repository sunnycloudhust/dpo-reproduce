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

Mặc định dùng `Qwen/Qwen2.5-0.5B-Instruct` và bộ preference toán local; đổi `model_name_or_path` trong `configs/dpo.yaml` hoặc truyền CLI:

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
- Đặt seed cố định khi so sánh các run.
- So sánh cùng split và cùng max sequence length.
- Theo dõi `loss`, `rewards/chosen`, `rewards/rejected`, `rewards/margins`.
- Dùng nhiều seed trước khi kết luận.
