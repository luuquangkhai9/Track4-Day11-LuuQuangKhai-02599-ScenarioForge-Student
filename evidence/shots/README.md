# Ảnh minh chứng — MSSV 2202602599

Các ảnh JPG là ảnh chụp thật từ viewer/index.html của lab, không phải ảnh dựng lại bằng công cụ sinh ảnh.

| File | Nguồn cảnh | Thời gian mô phỏng | Policy ego |
|---|---|---|---|
| bus-bypass.jpg | viewer.json, B-01 | 12.55 s (frame 251) | baseline |
| crosswalk-block.jpg | viewer.json, C-01 | 20.00 s (frame 400) | baseline |
| replay-only.jpg | replay.json, replay_only[0], scenario 0; replay_preview.json R-1 | 22.00 s | baseline, others=replay |
| hazard-bike.jpg | hazard.json, cảnh chính chương 10 | 7.50 s (frame 150) | baseline |

Ảnh hai kiểu thất bại được chụp trước khi có kết quả fix; lúc đó policy bóng mờ trùng baseline. Ảnh fix-ghost.jpg được chụp ở 12.55 s từ fix_comparison.json: ego dùng cutinLat=3.0 không va chạm, bóng mờ baseline va chạm ở cảnh B-01. Nhãn “chính sách của bạn” của trình xem chỉ là nhãn giao diện; policy thực tế phải đọc từ file dữ liệu của cảnh.

Ảnh drive-comparison.jpg: lượt điều khiển bằng bàn phím trong chương 10, ghi tại evidence/drive_scene.json, phanh -6 m/s² trong [5.50, 7.55) s; ảnh ở 7.50 s. Python phát lại không có sự kiện, hazard vẫn kích hoạt ở 5.45 s. Đã xem và thử lại cảnh; không phải thử phản xạ người thật.

