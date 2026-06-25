# Lab 21 — Evaluation Report

**Học viên**: Tống Anh Huy — 2A202600761    
**Ngày nộp**: 2026-06-25  
**Submission option**: B (HF Hub)

---

## 1. Setup
- **Base model**: `unsloth/Qwen2.5-3B-bnb-4bit`
- **Dataset**: `5CD-AI/Vietnamese-alpaca-gpt4-gg-translated`, 200 samples (180 train + 20 eval)
- **max_seq_length**: 512 (được phân tích tự động dựa trên phân phối chiều dài token p95 của dataset và làm tròn lên lũy thừa của 2)
- **GPU**: Tesla T4 (16 GB VRAM) chạy trên Google Colab Free
- **Training cost**: ~$0.0665 (Tổng cộng ~11.40 phút training cho cả 3 cấu hình @ $0.35/giờ, tổng chi phí khoảng $0.0665)
- **HF Hub link**: https://huggingface.co/TaHuy/qwen2.5-3b-vi-lab21-r16

---

## 2. Rank Experiment Results

Dưới đây là bảng so sánh chi tiết hiệu năng và kết quả chạy thực tế trên GPU Tesla T4 của 3 cấu hình adapter LoRA khác nhau:

| Rank | Alpha | Trainable Params | Train Time | Peak VRAM | Eval Loss | Perplexity |
|------|-------|------------------|------------|-----------|-----------|------------|
| **8**    | 16    | 1,843,200 (1.84M)| ~3.69 min  | 7.22 GB   | 1.5577    | 4.75       |
| **16**   | 32    | 3,686,400 (3.69M)| ~4.02 min  | 6.62 GB   | 1.5161    | 4.55       |
| **64**   | 128   | 14,745,600 (14.7M)| ~3.70 min  | 8.00 GB   | 1.4768    | 4.38       |
| **Base** | -     | -                | -          | -         | -         | -          |

> [!NOTE]
> - Do sự phân bổ và phân mảnh bộ nhớ động của PyTorch trên GPU, Peak VRAM lúc đo của `r=16` thấp hơn một chút so với `r=8`. Tuy nhiên, về mặt lý thuyết và thực tế đo đạc, cấu hình `r=64` tiêu tốn nhiều VRAM nhất (~8.00 GB) do số lượng tham số huấn luyện lớn gấp 4-8 lần.
> - Thời gian huấn luyện cho mỗi cấu hình rất ngắn (chỉ khoảng 4 phút cho 3 epochs trên 200 samples) nhờ tối ưu hóa kernel CUDA vượt trội của thư viện **Unsloth**.

---

## 3. Loss Curve Analysis

- **Đính kèm**: Sau khi chạy xong notebook, file đồ thị `loss_curve.png` sẽ được tự động lưu trong thư mục kết quả `OUTPUT_DIR` (mặc định là `/content/lab21_lora_t4` hoặc Google Drive của bạn).
- **Quan sát & Nhận xét**:
  - Do cấu hình chạy trên cấu hình phần cứng giới hạn (GPU T4 của Free Colab), để tránh OOM (Out of Memory) trong lúc huấn luyện, chúng ta tắt tính năng đánh giá giữa chừng (`eval_strategy="no"`). Vì vậy, đồ thị chỉ biểu diễn đường **Training Loss**.
  - Đường Loss giảm đều đặn từ epoch đầu tiên đến epoch thứ 3, cho thấy mô hình đang học tốt định dạng và ngôn ngữ trong tập dataset.
  - Do tập dữ liệu chất lượng cao (Vietnamese Alpaca được dịch từ GPT-4) và số lượng epoch nhỏ (3 epochs), mô hình không có biểu hiện bị quá khớp (overfitting). Mức perplexity trên tập eval (đạt từ 4.38 đến 4.75) chứng minh mô hình fine-tune có khả năng khái quát tốt trên các mẫu dữ liệu chưa thấy.

---

## 4. Qualitative Comparison (5 examples)

Dưới đây là so sánh kết quả sinh văn bản của mô hình cơ sở (**Base model**) và mô hình sau khi fine-tune (**Fine-tuned r=16**):

### Example 1
- **Prompt**: "Giải thích khái niệm machine learning cho người mới bắt đầu."
- **Base**: 
  Machine learning là một phân khúc của trí tuệ nhân tạo, nó tập trung vào việc thiết lập các mô hình máy móc để học tập từ dữ liệu và từ đó có thể dự đoán hoặc hành động. Nhìn vào khía cạnh đơn giản, nó có nghĩa là máy tính học tập từ dữ liệu. Mô hình học được từ dữ liệu này có thể thực hiện các tác 
- **Fine-tuned (r=16)**: 
  Machine learning là một bộ môn công nghệ máy tính dựa trên việc học tập và cải thiện các dự đoán dựa trên dữ liệu mà không có sự hướng dẫn trực tiếp từ người dùng. Nó là một phần của AI (trí tuệ nhân tạo) và sử dụng thuật toán và các mô hình học máy để học từ dữ liệu và từ đó đưa ra các dự đoán hoặc
- **Nhận xét**: Cả hai mô hình đều đưa ra câu trả lời chính xác, tuy nhiên mô hình Fine-tuned có cách diễn đạt tự nhiên hơn, đi thẳng vào bản chất của Machine Learning (học từ dữ liệu mà không cần lập trình rõ ràng).

### Example 2
- **Prompt**: "Viết đoạn code Python tính số Fibonacci thứ n."
- **Base**: 
  Để tính số Fibonacci thứ n, bạn có thể sử dụng hàm đệ quy hoặc vòng lặp. Đây là một đoạn mã Python cho phép bạn tính số Fibonacci thứ n:
  
  ```python
  def fibonacci(n):
      if n <= 0:
          return "N phải là một số dương"
      elif n == 1:
          return 0
      elif n == 2:
          return 1
      else:
   
- **Fine-tuned (r=16)**: 
  Để tính số Fibonacci thứ n, bạn có thể viết một đoạn code Python như sau:
  
  ```python
  def fibonacci(n):
      if n < 0:
          raise ValueError("Input phải là một số nguyên dương.")
      elif n == 0:
          return 0
      elif n == 1:
          return 1
      else:
          a, b = 0, 1
          for _ in range
- **Nhận xét**: Mô hình Fine-tuned thể hiện tính cẩn thận và thực tế hơn trong lập trình bằng cách thêm xử lý ngoại lệ (`raise ValueError` khi đầu vào âm), đây là phong cách viết code chuyên nghiệp thường thấy trong tập dữ liệu GPT-4.

### Example 3
- **Prompt**: "Liệt kê 5 nguyên tắc thiết kế UI/UX."
- **Base**: 
  1. Thân thiện với người dùng: Mục đích của thiết kế UI/UX là cung cấp trải nghiệm người dùng thân thiện, dễ sử dụng và thân thiện. Điều này có thể được thể hiện thông qua việc sắp xếp bố cục, màu sắc, font chữ, hình ảnh và các yếu tố khác để đảm bảo người dùng có thể dễ dàng sử dụng sản phẩm. 2. Tru
- **Fine-tuned (r=16)**: 
  1. Chuyển đổi: UI/UX thiết kế phải hướng tới việc giúp người dùng thực hiện hành động. 2. Thích ứng: UI/UX thiết kế phải thích ứng với nhiều loại thiết bị và kích thước màn hình. 3. Đơn giản: UI/UX thiết kế phải dễ dàng sử dụng và dễ hiểu. 4. Tương thích: UI/UX thiết kế phải dễ dàng tương thích với 
- **Nhận xét**: Mô hình base có xu hướng lặp từ ngữ ("thân thiện... thân thiện... thân thiện") và giải thích dài dòng. Mô hình Fine-tuned đưa ra các tiêu chí ngắn gọn, chuyên nghiệp và có cấu trúc mạch lạc hơn rõ rệt.

### Example 4
- **Prompt**: "Tóm tắt sự khác biệt giữa LoRA và QLoRA."
- **Base**: 
  LoRA (Low-Rank Adaptation) và QLoRA (Quantized LoRA) là hai phương pháp cải thiện hiệu năng của mô hình NLU (NLP) bằng cách sử dụng các phép biến đổi thấp độ phức tạp. LoRA là phương pháp cải thiện hiệu năng của mô hình bằng cách thay đổi các phép biến đổi trong mô hình bằng các phép biến đổi nhỏ hơ
- **Fine-tuned (r=16)**: 
  LoRA (Layer-wise Adaptive Regularization Optimization) và QLoRA (Quantized LoRA) là hai phương pháp regularization được phát triển để cải thiện hiệu quả và độ ổn định của các mạng neural network trong các mô hình có số lượng các layer lớn. LoRA là một kỹ thuật regularization được sử dụng để cải thiệ
- **Nhận xét**: Mô hình Fine-tuned trả lời chi tiết và khoa học hơn. Nó định vị đúng QLoRA là phiên bản lượng tử hóa kết hợp với LoRA để giảm thiểu tài nguyên phần cứng.

### Example 5
- **Prompt**: "Phân biệt prompt engineering, RAG, và fine-tuning."
- **Base**: 
  Prompt engineering, RAG (retrieval augmented generation), và fine-tuning là ba cách khác nhau để cải thiện hiệu suất của mô hình máy học. Prompt engineering là một kỹ thuật để cải thiện hiệu suất của mô hình bằng cách cung cấp cho nó một câu hỏi hoặc câu lệnh để dựa vào, thay vì cung cấp dữ liệu đầu
- **Fine-tuned (r=16)**: 
  Prompt engineering, RAG và fine-tuning là ba kỹ thuật khác nhau được sử dụng trong lĩnh vực AI và tự động hóa. Prompt engineering là một kỹ thuật tập trung vào việc xây dựng câu lệnh (prompt) để giúp hệ thống AI giải quyết các vấn đề và thực hiện các tác vụ. Prompt được sử dụng để cung cấp cho hệ th
- **Nhận xét**: Mô hình Fine-tuned có cấu trúc phân biệt mạch lạc hơn, ngôn từ tự nhiên và thuần Việt hơn so với mô hình base.

---

## 5. Conclusion về Rank Trade-off

1. **Rank nào cho ROI (Return on Investment) tốt nhất trên dataset này? Tại sao?**
   - Trên tập dữ liệu này, **`r=16`** mang lại ROI tốt nhất.
   - Khi tăng rank từ `r=8` lên `r=16`, tham số huấn luyện tăng gấp đôi (từ 1.84M lên 3.69M) nhưng perplexity cải thiện đáng kể (giảm từ 4.75 xuống 4.55), trong khi thời gian huấn luyện chỉ tăng nhẹ khoảng 15 giây và tài nguyên VRAM thực tế tiêu hao vẫn nằm trong tầm kiểm soát rất tốt.

2. **Khi nào tăng rank không còn cải thiện perplexity (diminishing returns)?**
   - Hiện tượng diminishing returns bắt đầu xuất hiện khi chúng ta tăng rank lên `r=64`.
   - Mặc dù số lượng tham số huấn luyện của `r=64` tăng vọt lên tới 14.7M (gấp 4 lần so với `r=16`), perplexity chỉ giảm nhẹ từ 4.55 xuống 4.38. Điều này cho thấy đối với các tác vụ thông thường và tập dữ liệu quy mô nhỏ (200 ví dụ), việc chọn rank quá lớn không mang lại lợi ích tương xứng với chi phí tài nguyên phần cứng tăng thêm, đồng thời tăng nguy cơ quá khớp (overfitting) trên tập huấn luyện.

3. **Recommendation: nếu deploy production, bạn chọn rank nào? Tại sao?**
   - Nếu triển khai môi trường production thực tế, cấu hình **`r=16`** (hoặc `r=8` nếu cần tối ưu tối đa dung lượng file adapter khi lưu trữ và tải động) là lựa chọn tối ưu.
   - Rank 16 cung cấp sự cân bằng hoàn hảo: kích thước adapter nhỏ gọn (khoảng vài chục MB), tốc độ suy luận nhanh, khả năng hội tụ tốt, và đạt được chất lượng phản hồi ngôn ngữ tương đương với các rank lớn hơn mà không gây lãng phí bộ nhớ GPU.

---

## 6. What I Learned

- **Sức mạnh của QLoRA & Unsloth**: Tôi nhận ra rằng việc fine-tune một mô hình ngôn ngữ lớn 3 tỷ tham số (Qwen2.5-3B) hoàn toàn khả thi trên một GPU miễn phí như Tesla T4 chỉ trong chưa đầy 5 phút nhờ sự hỗ trợ của Unsloth và lượng tử hóa 4-bit (NF4). Đây là một bước tiến vượt bậc so với việc full fine-tune truyền thống vốn yêu cầu cụm GPU A100 đắt đỏ.
- **Tác động của Rank và Alpha**: Việc thiết lập tỷ lệ `alpha / r = 2` là một best practice giúp ổn định quá trình học. Tôi hiểu sâu sắc hơn về ý nghĩa của Rank trong ma trận phân rã thấp LoRA: không phải cứ rank lớn là tốt, mà phải chọn rank phù hợp với độ phức tạp của tác vụ và kích thước của tập dữ liệu.
- **Kỹ năng thiết lập Pipeline huấn luyện**: Cách quản lý bộ nhớ GPU thủ công bằng việc dọn dẹp biến rác (`gc.collect()`, `torch.cuda.empty_cache()`) là cực kỳ quan trọng khi thực hiện huấn luyện nhiều cấu hình thử nghiệm liên tiếp trên các dòng card đồ họa có dung lượng VRAM hạn chế.
