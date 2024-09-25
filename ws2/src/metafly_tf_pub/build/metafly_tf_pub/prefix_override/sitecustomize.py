import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/blackandgold/ece569-fall2024/Lab2/ws2/src/metafly_tf_pub/install/metafly_tf_pub'
