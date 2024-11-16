import jetson.inference
import jetson.utils

net = jetson.inference.detectNet("ssd-mobilenet-v2", threshold=0.5)
camera = jetson.utils.videoSource("/dev/video0")  # '/dev/video0' for V4L2
display = jetson.utils.videoOutput("display://0")  # 'my_video.mp4' for file

while display.IsStreaming():
    img = camera.Capture()
    if img is None:  # capture timeout
        continue

    detections = net.Detect(img)

    # Print the detection results, each attribute on a new line
    for detection in detections:
        print(f"-- ClassID: {detection.ClassID}")
        print(f"-- Confidence: {detection.Confidence:.6f}")
        print(f"-- Left: {detection.Left:.5f}")
        print(f"-- Top: {detection.Top:.5f}")
        print(f"-- Right: {detection.Right:.5f}")
        print(f"-- Bottom: {detection.Bottom:.5f}")
        
        # 计算宽度、高度、面积和中心坐标
        width = detection.Right - detection.Left
        height = detection.Bottom - detection.Top
        area = width * height
        center_x = (detection.Left + detection.Right) / 2
        center_y = (detection.Top + detection.Bottom) / 2

        print(f"-- Width: {width:.6f}")
        print(f"-- Height: {height:.6f}")
        print(f"-- Area: {area:.2f}")
        print(f"-- Center: ({center_x:.3f}, {center_y:.3f})")
        print("")  # Print a blank line for better readability

    display.Render(img)
    display.SetStatus("Object Detection | Network {:.0f} FPS".format(net.GetNetworkFPS()))
