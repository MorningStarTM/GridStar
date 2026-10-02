"""
GridStar data-generation EC2 instance — one stack per episode range.

Config (set per-stack with `pulumi config set`):
    startEpisode   int     first episode (inclusive)
    endEpisode     int     last episode (exclusive)
    myIp           string  your IP in CIDR form, e.g. 203.189.189.170/32
    keyName        string  existing EC2 key pair name, e.g. gridstar-key
    hfToken        secret  HuggingFace write token

Usage:
    pulumi config set startEpisode 860
    pulumi config set endEpisode 900
    pulumi config set myIp 203.189.189.170/32
    pulumi config set keyName gridstar-key
    pulumi config set --secret hfToken hf_xxxxxxxxxxxx
    pulumi up
"""

import pulumi
import pulumi_aws as aws

config = pulumi.Config()
start_episode = config.require_int("startEpisode")
end_episode   = config.require_int("endEpisode")
my_ip         = config.require("myIp")
key_name      = config.require("keyName")
hf_token      = config.require_secret("hfToken")

STACK_TAG = f"gridstar-ep-{start_episode}-{end_episode}"

# ── Security group — SSH + dashboard, both scoped to your IP only ────────────
sg = aws.ec2.SecurityGroup(
    f"{STACK_TAG}-sg",
    description="GridStar data generation - SSH + monitor dashboard",
    ingress=[
        aws.ec2.SecurityGroupIngressArgs(
            protocol="tcp", from_port=22, to_port=22, cidr_blocks=[my_ip],
            description="SSH",
        ),
        aws.ec2.SecurityGroupIngressArgs(
            protocol="tcp", from_port=8000, to_port=8000, cidr_blocks=[my_ip],
            description="Monitor dashboard",
        ),
    ],
    egress=[
        aws.ec2.SecurityGroupEgressArgs(
            protocol="-1", from_port=0, to_port=0, cidr_blocks=["0.0.0.0/0"],
        ),
    ],
    tags={"Name": f"{STACK_TAG}-sg"},
)

# ── Latest Ubuntu 26.04 x86_64 AMI — avoids repeating the manual mis-picks ───
ami = aws.ec2.get_ami(
    most_recent=True,
    owners=["099720109477"],  # Canonical
    filters=[
        aws.ec2.GetAmiFilterArgs(
            name="name", values=["ubuntu/images/hvm-ssd*/ubuntu-*-amd64-server-*"],
        ),
        aws.ec2.GetAmiFilterArgs(name="architecture", values=["x86_64"]),
        aws.ec2.GetAmiFilterArgs(name="virtualization-type", values=["hvm"]),
    ],
)

# ── User-data: bakes in every fix learned the hard way on the manual instance
#    - Python 3.11 via deadsnakes (Ubuntu 26.04 ships 3.14, too new for torch/numpy pins)
#    - torch installed from the CPU-only index (avoids the 554MB CUDA bundle)
#    - episode range injected via sed into data_main.py
#    - both data_main.py and monitor_server.py launched unattended on boot
user_data = pulumi.Output.all(hf_token).apply(lambda args: f"""#!/bin/bash
set -ex
exec > /var/log/user-data.log 2>&1

apt-get update -y
add-apt-repository ppa:deadsnakes/ppa -y
apt-get update -y
apt-get install -y python3.11 python3.11-venv python3.11-dev git

git clone https://github.com/MorningStarTM/GridStar.git /home/ubuntu/GridStar
cd /home/ubuntu/GridStar

python3.11 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

sed -i -E "s/start_episode=[0-9]+, end_episode=[0-9]+/start_episode={start_episode}, end_episode={end_episode}/" data_main.py

chown -R ubuntu:ubuntu /home/ubuntu/GridStar

cat > /home/ubuntu/GridStar/.env <<EOF
export HF_TOKEN={args[0]}
EOF
chown ubuntu:ubuntu /home/ubuntu/GridStar/.env

su - ubuntu -c "cd /home/ubuntu/GridStar && source .env && source venv/bin/activate && nohup python data_main.py > generation.log 2>&1 &"
su - ubuntu -c "cd /home/ubuntu/GridStar && source venv/bin/activate && nohup python monitor_server.py > monitor.log 2>&1 &"
""")

# ── EC2 instance — 30GB root volume (delete_after_push keeps disk usage flat,
#    so no need for the bigger volume used before that existed)
instance = aws.ec2.Instance(
    STACK_TAG,
    instance_type="t3.small",
    ami=ami.id,
    key_name=key_name,
    vpc_security_group_ids=[sg.id],
    user_data=user_data,
    user_data_replace_on_change=True,
    root_block_device=aws.ec2.InstanceRootBlockDeviceArgs(
        volume_size=30, volume_type="gp3",
    ),
    tags={"Name": STACK_TAG},
)

pulumi.export("public_ip", instance.public_ip)
pulumi.export("dashboard_url", instance.public_ip.apply(lambda ip: f"http://{ip}:8000"))
pulumi.export("ssh_command", instance.public_ip.apply(
    lambda ip: f'ssh -i "{key_name}.pem" ubuntu@{ip}'
))
