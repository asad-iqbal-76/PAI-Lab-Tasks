import cv2
import numpy as np
import folium
import os
import geocoder

# Load YOLO model
net = cv2.dnn.readNet("yolo/yolov3.weights", "yolo/yolov3.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i-1] for i in net.getUnconnectedOutLayers()]

# Load class names
with open("yolo/coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Animal classes to detect
ANIMAL_CLASSES = ["sheep", "cow", "horse", "bear", "zebra", "giraffe", "elephant"]

def get_current_location():
    """Get current location using IP address or GPSD"""
    try:
        # Try GPSD first (for hardware GPS)
        import gpsd
        gpsd.connect()
        packet = gpsd.get_current()
        return packet.lat, packet.lon
    except:
        # Fallback to IP-based location
        g = geocoder.ip('me')
        if g.latlng:
            return g.latlng[0], g.latlng[1]
        else:
            # Final fallback to dummy coordinates
            return 51.5074, -0.1278  # London coordinates

def create_alert_map(lat, lon, animal, count):
    """Create interactive map with alert marker"""
    m = folium.Map(location=[lat, lon], zoom_start=14)
    
    # Alert marker with popup
    folium.Marker(
        [lat, lon],
        popup=f"🛑 HERD ALERT\n{count} {animal}s",
        icon=folium.Icon(color='red', icon='exclamation-triangle')
    ).add_to(m)
    
    # Add current location marker
    folium.Marker(
        [lat, lon],
        popup="Your Location",
        icon=folium.Icon(color='blue', icon='user')
    ).add_to(m)
    
    # Add satellite tile layer
    folium.TileLayer(
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attr='Esri',
        name='Satellite',
        overlay=False,
        control=True
    ).add_to(m)
    
    m.save("map_alert.html")
    print(f"Map alert saved with coordinates: {lat:.4f}, {lon:.4f}")

def detect_herd(image_path):
    """Main detection function with real-time location"""
    # Get current location
    lat, lon = get_current_location()
    print(f"Current location: {lat:.4f}, {lon:.4f}")
    
    # Load and verify image
    if not os.path.exists(image_path):
        print(f"Error: File '{image_path}' not found")
        return
    
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Unable to load image at {image_path}")
        return

    # Detect objects
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Process detections
    animal_counts = {}
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and classes[class_id] in ANIMAL_CLASSES:
                animal_type = classes[class_id]
                animal_counts[animal_type] = animal_counts.get(animal_type, 0) + 1

    # Check for herds
    alert_triggered = False
    for animal, count in animal_counts.items():
        if count >= 3:
            print(f"HERD ALERT: {count} {animal}s detected!")
            create_alert_map(lat, lon, animal, count)
            alert_triggered = True

    if not alert_triggered:
        print("No herds detected")
    
    # Show image with detections
    cv2.imshow("Detection Results", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Test with your image
image_path = r"C:\Users\Asaad\Desktop\Semester 4\PAI (Lab)\Task # 6\animal_herd_detection\sheep_herd.jpg"
if not os.path.exists(image_path):
    print(f"Error: File '{image_path}' does not exist.")
else:
    print(f"File '{image_path}' found.")
    detect_herd(image_path)