import serial
import struct

ser = serial.Serial("/dev/ttyTHS1", baudrate=115200, timeout=100000)

test = struct.pack("!ii", 1500, 1500)
ser.write(test)

while True:
    values = input("DIR|PROPU ").strip().split("|")
    test = struct.pack("!ii", int(values[0]), int(values[1]))
    print(ser.write(test))
