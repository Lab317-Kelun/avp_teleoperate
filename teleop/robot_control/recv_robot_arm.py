import lcm
from msg_pkg import msg

def handler(channel, data):
    message = msg.decode(data)
    print(f"\nReceived on channel '{channel}':")
    print(message)

def main():
    lc = lcm.LCM()
    lc.subscribe("arm_action", handler)
    print("Listening for messages on 'upper_data'...")

    try:
        while True:
            lc.handle()  # 阻塞等待消息
    except KeyboardInterrupt:
        print("Exiting.")

if __name__ == "__main__":
    main()