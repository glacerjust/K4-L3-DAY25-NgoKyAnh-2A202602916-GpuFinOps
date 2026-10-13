# Bài Viết Phân Tích Kỹ Thuật: GPU FinOps Optimization (NimbusAI)

> **Học viên:** Ngô Kỳ Anh 
> **Mã học viên / MSSV:** 2A202602916 
> **Khóa học:** AICB · Phase 2 · Track 2 (Infrastructure) · Day 25  
> **Mục tiêu:** Cắt giảm chi phí GPU 40–95%, chuẩn hoá đơn vị đo `$/1M-token`, phân bổ trách nhiệm và tối ưu tính bền vững (Sustainability).

---

## 1. Baseline vs. Optimized Spend & Đơn vị `$/1M-Token`

### 1.1 Tổng quan chi phí
- **Baseline Spend (Hàng tháng):** $27,133 / tháng (~$904.43 / ngày)
- **Optimized Spend (Hàng tháng):** $14,626 / tháng (~$487.53 / ngày)
- **Tổng tiết kiệm dự kiến:** **$12,507 / tháng (Giảm 46.1%)**

### 1.2 Đơn vị đo lường: Tại sao `$/1M-token` thay thế `$/GPU-hr`?
Trong kỷ nguyên LLM, `$/GPU-hr` chỉ phản ánh số tiền doanh nghiệp trả để thuê phần cứng trong một khoảng thời gian, hoàn toàn không phản ánh **giá trị thực hoặc sản lượng token** thu được:
- Hai đội ngũ kỹ thuật có thể cùng thuê GPU H100 với giá $2.50/giờ, nhưng đội tối ưu tốt (dùng model routing, caching, batching) có thể phục vụ 10× số token với cùng 1 giờ GPU đó.
- Tại NimbusAI, trước khi tối ưu: chi phí inference ở mức **$36.643 / 1M-token** (mô hình lớn chạy naive, không cache, không batch).
- Sau khi tối ưu: chi phí inference giảm xuống chỉ còn **$10.374 / 1M-token** — **tiết kiệm 71.7%** đơn giá token.
- Đo bằng `$/1M-token` liên kết trực tiếp chi phí hạ tầng với doanh thu (COGS) và mức độ tiêu thụ của người dùng cuối.

---

## 2. Phân Tích Chi Tiết Từng Đòn Bẩy Tiết Kiệm (FinOps Levers)

| Đòn bẩy | Tiết kiệm (USD/tháng) | Tỷ trọng | Cơ chế hoạt động & Đánh giá |
|---|---|---|---|
| **Purchasing Strategy (Spot / Reserved)** | **$10,040** | **80.3%** | **Đóng góp lớn nhất.** Chuyển đổi các job 24/7 sang Reserved (chiết khấu 45%) và training jobs sang Spot + Checkpoint (chiết khấu >40%). |
| **Inference Levers (Cascade / Cache / Batch)** | **$1,212** | **9.7%** | Tối ưu đường ống serving: Cascade sang model nhỏ (80% requests), cache 90% input prefix trùng lặp, batching giảm 50% chi phí. |
| **Right-size "GPU-Util Lies"** | **$655** | **5.2%** | Nhận diện GPU bị nghẽn băng thông memory-bound và hạ cấp từ H100 xuống A100/A10G mà không ảnh hưởng latency/throughput. |
| **Kill Idle GPUs** | **$600** | **4.8%** | Tự động tắt các GPU chạy không (idle utilization < 10%) qua đêm (như `gpu-h100-5`). |
| **Tổng cộng** | **$12,507** | **100.0%** | |

### Tại sao Purchasing Strategy đóng góp nhiều nhất?
Hạ tầng training và fine-tuning tiêu tốn số giờ GPU khổng lồ liên tục. Khi chuyển từ On-Demand ($2.50/h cho H100) sang Spot ($1.50/h) hoặc Reserved ($1.40/h), khoản chiết khấu áp trực tiếp lên hàng nghìn giờ GPU tích luỹ, tạo ra mức ROI tiền mặt ngay lập tức lớn nhất trong tổ chức.

---

## 3. "GPU-Util Lie" — Bản Chất Kỹ Thuật và Tác Động Tài Chính

### 3.1 Cơ chế kỹ thuật của "Lie"
- Công cụ `nvidia-smi` báo cáo chỉ số `GPU-Util %` dựa trên tỷ lệ thời gian có ít nhất một kernel đang hoạt động trên GPU trong chu kỳ lấy mẫu.
- **`GPU-Util` KHÔNG ĐO Tensor Core occupancy hay số phép tính FLOPs thực tế.**
- Trong dữ liệu telemetry của NimbusAI, `gpu-h100-4` hiển thị **98% GPU-Util** nhưng **MFU (Model FLOPs Utilization) chỉ đạt 20.2%**.
- **Nguyên nhân cốt lõi:**
  1. *Memory bandwidth stalls:* GPU liên tục chờ nạp dữ liệu từ HBM/DRAM thay vì thực hiện tính toán ma trận (thường gặp ở khâu autoregressive decode trong LLM với batch size nhỏ).
  2. *Kernel launch overhead & small tensors:* Các kernel quá nhỏ, không bão hoà được SM (Streaming Multiprocessor).
  3. *Synchronization barriers:* Chờ trao đổi gradients giữa các GPU.

### 3.2 Tác động tài chính
Doanh nghiệp đang trả $2.50/giờ cho một "con quái vật" H100 990 TFLOPs nhưng chỉ thực sự thu được ~200 TFLOPs (tương đương năng lực một GPU A100 $1.79/h hoặc A10G $1.00/h). Việc trả thừa này gây lãng phí ít nhất **$0.71–$1.50 mỗi giờ cho mỗi GPU** bị "lie".

---

## 4. Báo Cáo Các Phần Mở Rộng Đã Thực Hiện ("Your Turn" Extensions)

Chúng tôi đã triển khai mở rộng 4/5 extensions trực tiếp vào code engine và missions:

### 4.1 Extension 1: Nâng cấp chính sách `recommend_tier()` (`finops/pricing.py`)
- **Triển khai:** Thêm tham số `job_days`, `gpu_type`, và `interruption_rate`.
- **Logic mới:** 
  - Không cam kết Reserved dài hạn (1–3 năm) cho các job thử nghiệm ngắn ngày (`job_days < 15`) để tránh khóa dòng tiền.
  - Tự động fallback về Reserved/On-Demand nếu tỷ lệ gián đoạn của loại GPU đó vượt ngưỡng chịu đựng (>15%).
- **Insight:** Chính sách linh hoạt giúp bảo vệ vốn lưu động của startup thay vì chỉ tối ưu lý thuyết trên giấy.

### 4.2 Extension 2: Right-sizing theo MBU & `$/GB-VRAM` (`missions/m1_efficiency_audit.py`)
- **Triển khai:** Tính toán ma trận `$/GB-VRAM` và băng thông đỉnh `peak_bw_tbs` của từng GPU trong catalog.
- **Kết quả đo lường:**
  - `gpu-h100-4` (H100) có achieved bandwidth là 1.51 TB/s (MBU 45%).
  - GPU A100 cung cấp 2.0 TB/s peak bandwidth và 80GB HBM với giá $1.79/hr.
  - Thay thế H100 bằng A100 đáp ứng 100% nhu cầu băng thông của workload decode, tiết kiệm ngay **$0.71/hr (~$511/tháng, giảm 28.4%)**.

### 4.3 Extension 3: Kinh tế học Prompt Caching `cache_is_worth_it()` (`finops/pricing.py`)
- **Triển khai:** Xây dựng hàm kiểm định break-even điểm hòa vốn giữa chi phí ghi cache (write cost $3.75/1M) và tiết kiệm đọc (read discount 90%).
- **Kết quả:** Ngưỡng hòa vốn cho model lớn là **1.38 lần đọc**. Với tỷ lệ tái sử dụng prompt trung bình ~3.5 lần trong hệ thống chat/RAG của NimbusAI, `cache_is_worth_it()` trả về `True`, xác nhận việc kích hoạt prompt caching mang lại lợi nhuận ròng.

### 4.4 Extension 4: Phân tích ngân sách Reasoning Traffic & "Energy Bomb" (`missions/m2_inference_levers.py`)
- **Kết quả đo lường:**
  - Traffic `is_reasoning=1` chỉ chiếm **~8.3% tổng số requests** (199/2,400 requests).
  - Tuy nhiên, do sinh thêm chuỗi suy luận dài (chain-of-thought) và hệ số năng lượng 80×, traffic này tiêu thụ tới **87.9% tổng điện năng serving** (43.4 kWh / 49.4 kWh)!
- **Đề xuất routing:** Chỉ kích hoạt reasoning mode khi bài toán yêu cầu logic phức tạp (code/math) hoặc khi phân loại độ khó vượt ngưỡng tin cậy.

### 4.5 Extension 5: Carbon-Aware Scheduling (`missions/m3_purchasing.py`)
- **Kết quả đo lường:**
  - Chuyển các job training/finetune có thể gián đoạn (`interruptible=1`, tổng 32,880 kWh/tháng) từ `us-east-1` (380 gCO2/kWh) sang vùng điện tái tạo `europe-north1` (30 gCO2/kWh).
  - Giảm phát thải **11,508 kg CO2e / tháng (giảm 92.1% lượng phát thải carbon)** mà không phát sinh thêm chi phí phần cứng.

---

## 5. Khuyến Nghị Hành Động Cho Ban Lãnh Đạo NimbusAI

Nếu được giao trọng trách FinOps Lead tại NimbusAI, 3 hành động tiên quyết trong 60 ngày đầu:

1. **Hành động 1 (Tuần 1 - Thu hồi vốn tức thì): Thiết lập Auto-Shutdown & Cascade Routing Mặc Định**
   - Thiết lập cron script tắt instance khi idle > 15 phút (tiết kiệm ngay $600/tháng).
   - Đặt classifier proxy (LiteLLM) mặc định định tuyến 80% prompt đơn giản sang Small Model Tier, cắt giảm 70% chi phí serving.
2. **Hành động 2 (Tháng 1 - Tối ưu danh mục cam kết): Mua Reserved cho 24/7 Inference & Chuyển Training sang Spot**
   - Ký cam kết Reserved 3 năm cho các dịch vụ cốt lõi chạy 24/7 (assistant, RAG), khóa mức chiết khấu 45%.
   - Xây dựng framework lưu checkpoint tự động mỗi 30 phút lên object storage để an tâm vận hành training trên Spot instances.
3. **Hành động 3 (Tháng 2 - Xây dựng văn hóa FinOps): Kích hoạt Chargeback tự động với chuẩn FOCUS**
   - Tận dụng tag coverage hiện tại (92% > 80%), xuất dữ liệu chi phí chuẩn FOCUS hàng tháng và phân bổ thẳng về ngân sách của từng Product/Research Lead.
   - Khi kỹ sư nhìn thấy chi phí của chính đội ngũ mình (Showback/Chargeback), họ sẽ tự giác chọn batch API và tối ưu prompt.
