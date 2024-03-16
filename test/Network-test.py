import socket
import struct

def main():
    # Define the multicast group and port to listen on
    multicast_group = '224.16.16.16'
    server_address = ('', 33333)  # Listen on all available interfaces

    # Create the socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)

    # Allow multiple copies of this program on one machine
    # (i.e., allow the socket to be bound to an address that is already in use)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    # Bind to the server address
    sock.bind(server_address)

    # Tell the operating system to add the socket to the multicast group
    # on all interfaces.
    group = socket.inet_aton(multicast_group)
    mreq = struct.pack('4sL', group, socket.INADDR_ANY)
    sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)

    # Receive/respond loop
    print('Waiting to receive message')
    while True:
        data, address = sock.recvfrom(1024)  # Buffer size is 1024 bytes
        print(f'Received {len(data)} bytes from {address}')
        print(data)

        # Optional: send a response to the sender
        # sock.sendto(b'Acknowledged', address)

if __name__ == '__main__':
    main()
