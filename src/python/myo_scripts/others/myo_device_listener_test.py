import myo
import time

def main():
  myo.init(sdk_path='/Users/vaynee_sungeelee/Desktop/mab_classifier/myo-sdk/')
  hub = myo.Hub()
  listener = myo.ApiDeviceListener()
  with hub.run_in_background(listener.on_event):
    print("Waiting for a Myo to connect ...")
    device = listener.wait_for_single_device(2)
    if not device:
      print("No Myo connected after 2 seconds.")
      return
    print("Hello, Myo! Requesting RSSI ...")
    device.request_rssi()
    while hub.running and device.connected and not device.rssi:
      print("Waiting for RRSI...")
      time.sleep(0.001)
    print("RSSI:", device.rssi)
    print("Goodbye, Myo!")