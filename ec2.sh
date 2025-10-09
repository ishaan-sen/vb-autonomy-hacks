#!/bin/bash

# EC2 Instance Creation and SSH Setup Script
# This script creates an EC2 instance and configures easy SSH access

set -e  # Exit on error

# Configuration variables
INSTANCE_NAME="my-dev-instance"
INSTANCE_TYPE="t2.micro"  # Free tier eligible
AMI_ID=""  # Will be auto-detected
KEY_NAME="my-ec2-key"
SECURITY_GROUP_NAME="my-ssh-sg"
REGION="us-east-1"  # Change to your preferred region

echo "=========================================="
echo "EC2 Instance Setup Script"
echo "=========================================="

# Step 1: Create SSH key pair if it doesn't exist
echo -e "\n[1/6] Creating SSH key pair..."
if [ ! -f ~/.ssh/${KEY_NAME}.pem ]; then
    aws ec2 create-key-pair \
        --key-name ${KEY_NAME} \
        --region ${REGION} \
        --query 'KeyMaterial' \
        --output text > ~/.ssh/${KEY_NAME}.pem
    
    chmod 400 ~/.ssh/${KEY_NAME}.pem
    echo "✓ Key pair created: ~/.ssh/${KEY_NAME}.pem"
else
    echo "✓ Key pair already exists: ~/.ssh/${KEY_NAME}.pem"
fi

# Step 2: Get default VPC ID
echo -e "\n[2/6] Getting default VPC..."
VPC_ID=$(aws ec2 describe-vpcs \
    --region ${REGION} \
    --filters "Name=isDefault,Values=true" \
    --query 'Vpcs[0].VpcId' \
    --output text)

if [ "$VPC_ID" == "None" ] || [ -z "$VPC_ID" ]; then
    echo "❌ No default VPC found. Please create one first."
    exit 1
fi
echo "✓ Using VPC: ${VPC_ID}"

# Step 2.5: Auto-detect AMI if not specified
if [ -z "$AMI_ID" ]; then
    echo -e "\n[2.5/6] Auto-detecting latest Amazon Linux 2023 AMI..."
    AMI_ID=$(aws ec2 describe-images \
        --region ${REGION} \
        --owners amazon \
        --filters "Name=name,Values=al2023-ami-2023.*-x86_64" "Name=state,Values=available" \
        --query 'Images | sort_by(@, &CreationDate) | [-1].ImageId' \
        --output text)
    
    if [ "$AMI_ID" == "None" ] || [ -z "$AMI_ID" ]; then
        echo "❌ Could not find Amazon Linux 2023 AMI. Trying Ubuntu..."
        AMI_ID=$(aws ec2 describe-images \
            --region ${REGION} \
            --owners 099720109477 \
            --filters "Name=name,Values=ubuntu/images/hvm-ssd/ubuntu-jammy-22.04-amd64-server-*" "Name=state,Values=available" \
            --query 'Images | sort_by(@, &CreationDate) | [-1].ImageId' \
            --output text)
    fi
    
    if [ "$AMI_ID" == "None" ] || [ -z "$AMI_ID" ]; then
        echo "❌ Could not auto-detect AMI. Please specify AMI_ID manually."
        exit 1
    fi
    
    echo "✓ Using AMI: ${AMI_ID}"
fi

# Step 3: Create security group (if it doesn't exist)
echo -e "\n[3/6] Setting up security group..."
SG_ID=$(aws ec2 describe-security-groups \
    --region ${REGION} \
    --filters "Name=group-name,Values=${SECURITY_GROUP_NAME}" "Name=vpc-id,Values=${VPC_ID}" \
    --query 'SecurityGroups[0].GroupId' \
    --output text 2>/dev/null || echo "None")

if [ "$SG_ID" == "None" ] || [ -z "$SG_ID" ]; then
    # Create security group
    SG_ID=$(aws ec2 create-security-group \
        --group-name ${SECURITY_GROUP_NAME} \
        --description "Security group for SSH access" \
        --vpc-id ${VPC_ID} \
        --region ${REGION} \
        --query 'GroupId' \
        --output text)
    
    # Get your public IP
    MY_IP=$(curl -s https://checkip.amazonaws.com)
    
    # Add SSH rule (port 22) from your IP only
    aws ec2 authorize-security-group-ingress \
        --group-id ${SG_ID} \
        --protocol tcp \
        --port 22 \
        --cidr ${MY_IP}/32 \
        --region ${REGION}
    
    echo "✓ Security group created: ${SG_ID}"
    echo "  SSH access allowed from: ${MY_IP}"
else
    echo "✓ Using existing security group: ${SG_ID}"
fi

# Step 4: Launch EC2 instance
echo -e "\n[4/6] Launching EC2 instance..."
INSTANCE_ID=$(aws ec2 run-instances \
    --image-id ${AMI_ID} \
    --instance-type ${INSTANCE_TYPE} \
    --key-name ${KEY_NAME} \
    --security-group-ids ${SG_ID} \
    --region ${REGION} \
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=${INSTANCE_NAME}}]" \
    --query 'Instances[0].InstanceId' \
    --output text)

echo "✓ Instance launched: ${INSTANCE_ID}"

# Step 5: Wait for instance to be running
echo -e "\n[5/6] Waiting for instance to start..."
aws ec2 wait instance-running \
    --instance-ids ${INSTANCE_ID} \
    --region ${REGION}

# Get public IP
PUBLIC_IP=$(aws ec2 describe-instances \
    --instance-ids ${INSTANCE_ID} \
    --region ${REGION} \
    --query 'Reservations[0].Instances[0].PublicIpAddress' \
    --output text)

echo "✓ Instance is running!"
echo "  Public IP: ${PUBLIC_IP}"

# Step 6: Configure SSH config for easy access
echo -e "\n[6/6] Configuring SSH..."

# Create SSH config entry
SSH_CONFIG_ENTRY="
# ${INSTANCE_NAME} - Created $(date)
Host ${INSTANCE_NAME}
    HostName ${PUBLIC_IP}
    User ec2-user
    IdentityFile ~/.ssh/${KEY_NAME}.pem
    StrictHostKeyChecking no
    UserKnownHostsFile=/dev/null
"

# Add to SSH config if not already present
if ! grep -q "Host ${INSTANCE_NAME}" ~/.ssh/config 2>/dev/null; then
    echo "${SSH_CONFIG_ENTRY}" >> ~/.ssh/config
    echo "✓ SSH config updated"
else
    echo "✓ SSH config entry already exists"
fi

# Step 7: Create helper scripts
echo -e "\n[7/7] Creating helper scripts..."

# Create connection script
cat > ~/connect-${INSTANCE_NAME}.sh << EOF
#!/bin/bash
# Quick connect to ${INSTANCE_NAME}
ssh ${INSTANCE_NAME}
EOF
chmod +x ~/connect-${INSTANCE_NAME}.sh

# Create status check script
cat > ~/status-${INSTANCE_NAME}.sh << EOF
#!/bin/bash
# Check status of ${INSTANCE_NAME}
aws ec2 describe-instances \\
    --instance-ids ${INSTANCE_ID} \\
    --region ${REGION} \\
    --query 'Reservations[0].Instances[0].[InstanceId,State.Name,PublicIpAddress,InstanceType]' \\
    --output table
EOF
chmod +x ~/status-${INSTANCE_NAME}.sh

# Create stop script
cat > ~/stop-${INSTANCE_NAME}.sh << EOF
#!/bin/bash
# Stop ${INSTANCE_NAME}
echo "Stopping instance ${INSTANCE_ID}..."
aws ec2 stop-instances --instance-ids ${INSTANCE_ID} --region ${REGION}
echo "Instance stopped. Run start-${INSTANCE_NAME}.sh to start it again."
EOF
chmod +x ~/stop-${INSTANCE_NAME}.sh

# Create start script
cat > ~/start-${INSTANCE_NAME}.sh << EOF
#!/bin/bash
# Start ${INSTANCE_NAME}
echo "Starting instance ${INSTANCE_ID}..."
aws ec2 start-instances --instance-ids ${INSTANCE_ID} --region ${REGION}
aws ec2 wait instance-running --instance-ids ${INSTANCE_ID} --region ${REGION}

# Get new public IP
NEW_IP=\$(aws ec2 describe-instances \\
    --instance-ids ${INSTANCE_ID} \\
    --region ${REGION} \\
    --query 'Reservations[0].Instances[0].PublicIpAddress' \\
    --output text)

echo "Instance started! New IP: \${NEW_IP}"
echo "Updating SSH config..."

# Update SSH config with new IP
sed -i.bak "s/HostName .*/HostName \${NEW_IP}/" ~/.ssh/config

echo "✓ Ready to connect: ssh ${INSTANCE_NAME}"
EOF
chmod +x ~/start-${INSTANCE_NAME}.sh

# Create terminate script
cat > ~/terminate-${INSTANCE_NAME}.sh << EOF
#!/bin/bash
# Terminate ${INSTANCE_NAME} (WARNING: This deletes the instance!)
read -p "Are you sure you want to TERMINATE instance ${INSTANCE_ID}? (yes/no): " confirm
if [ "\$confirm" == "yes" ]; then
    echo "Terminating instance..."
    aws ec2 terminate-instances --instance-ids ${INSTANCE_ID} --region ${REGION}
    echo "Instance terminated."
    # Clean up SSH config
    sed -i.bak "/# ${INSTANCE_NAME}/,/UserKnownHostsFile/d" ~/.ssh/config
    echo "SSH config cleaned up."
else
    echo "Termination cancelled."
fi
EOF
chmod +x ~/terminate-${INSTANCE_NAME}.sh

echo "✓ Helper scripts created in home directory"

# Final summary
echo -e "\n=========================================="
echo "✓ SETUP COMPLETE!"
echo "=========================================="
echo -e "\nInstance Details:"
echo "  Instance ID: ${INSTANCE_ID}"
echo "  Instance Name: ${INSTANCE_NAME}"
echo "  Public IP: ${PUBLIC_IP}"
echo "  Region: ${REGION}"
echo -e "\nQuick Commands:"
echo "  Connect:    ssh ${INSTANCE_NAME}"
echo "  Or run:     ~/connect-${INSTANCE_NAME}.sh"
echo "  Status:     ~/status-${INSTANCE_NAME}.sh"
echo "  Stop:       ~/stop-${INSTANCE_NAME}.sh"
echo "  Start:      ~/start-${INSTANCE_NAME}.sh"
echo "  Terminate:  ~/terminate-${INSTANCE_NAME}.sh"
echo -e "\nWaiting 30 seconds for instance to fully initialize..."
sleep 30
echo -e "\n✓ Ready to connect! Try: ssh ${INSTANCE_NAME}"
