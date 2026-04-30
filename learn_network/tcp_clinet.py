import socket
import struct
import json
import cv2
import numpy as np

# ------------------ 配置 ------------------
LISTEN_IP = "0.0.0.0"          # 监听所有本机 IP
LISTEN_PORT = 8888
# ------------------------------------------

def recv_exact(conn, num_bytes):
    """保证接收指定字节数"""
    data = b''
    while len(data) < num_bytes:
        packet = conn.recv(num_bytes - len(data))
        if not packet:
            return None
        data += packet
    return data

def main():
    # 创建 TCP 服务端
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((LISTEN_IP, LISTEN_PORT))
    server_socket.listen(1)
    print(f"等待树莓派连接... 监听端口 {LISTEN_PORT}")

    conn, addr = server_socket.accept()
    print(f"连接成功，客户端地址: {addr}")

    try:
        while True:
            # 1. 接收 4 字节的 JSON 长度
            header = recv_exact(conn, 4)
            if header is None:
                print("连接断开")
                break
            json_len = struct.unpack('>I', header)[0]

            # 2. 接收 JSON 数据
            json_bytes = recv_exact(conn, json_len)
            if json_bytes is None:
                break
            data = json.loads(json_bytes.decode('utf-8'))

            # TODO: 根据 data 里的 boxes 等信息做你自己的处理
            # 例如：打印人数，触发警报，或传递给 AI 模型分析
            person_count = len(data.get("boxes", []))
            if person_count > 0:
                print(f"检测到 {person_count} 个人")

            # 3. 接收图像数据
            #    这里采用超时方式接收全部图像字节（会一直收到连接超时或断开）
            conn.settimeout(1.0)
            img_data = b''
            try:
                while True:
                    chunk = conn.recv(4096)
                    if not chunk:
                        break
                    img_data += chunk
            except socket.timeout:
                pass                      # 超时视为一帧接收完毕
            conn.settimeout(None)

            if not img_data:
                continue

            # 4. 解码图像并在上面绘制检测框
            img_array = np.frombuffer(img_data, dtype=np.uint8)
            frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

            if frame is not None:
                # 绘制所有框
                for box in data.get("boxes", []):
                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    # TODO: 可根据 data["classes"] 添加标签文字

                # 显示
                cv2.imshow("Real-Time Detection", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    except KeyboardInterrupt:
        print("PC 服务端退出")
    finally:
        conn.close()
        server_socket.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
