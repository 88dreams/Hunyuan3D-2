# S3 Upload Access for Third Parties

This guide explains how to create a dedicated IAM user with limited permissions to allow a third party to upload files to your S3 bucket.

## Overview

We'll create an IAM user that can **only upload files** to a specific folder in your S3 bucket. They won't be able to:
- List bucket contents
- Download files
- Delete files
- Access other AWS services

---

## Step 1: Create an IAM Policy

1. Log in to the [AWS Console](https://console.aws.amazon.com/)
2. Go to **IAM** (Identity and Access Management)
3. Click **Policies** in the left sidebar
4. Click **Create policy**
5. Select the **JSON** tab
6. Paste the following policy:

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Sid": "AllowUploadOnly",
            "Effect": "Allow",
            "Action": [
                "s3:PutObject"
            ],
            "Resource": "arn:aws:s3:::YOUR-BUCKET-NAME/uploads/*"
        }
    ]
}
```

> **Important:** Replace `YOUR-BUCKET-NAME` with your actual bucket name.
> 
> The `/uploads/*` path restricts uploads to the `uploads/` folder. Modify this path as needed.

7. Click **Next**
8. Name the policy: `S3-Upload-Only-Policy`
9. Add a description: "Allows uploading files to the uploads folder only"
10. Click **Create policy**

---

## Step 2: Create an IAM User

1. In IAM, click **Users** in the left sidebar
2. Click **Create user**
3. Enter a username: `upload-user` (or a descriptive name for the third party)
4. Click **Next**
5. Select **Attach policies directly**
6. Search for and select `S3-Upload-Only-Policy` (the policy you just created)
7. Click **Next**
8. Click **Create user**

---

## Step 3: Create Access Keys

1. Click on the user you just created
2. Go to the **Security credentials** tab
3. Scroll down to **Access keys**
4. Click **Create access key**
5. Select **Third-party service** (or **Application running outside AWS**)
6. Check the confirmation box and click **Next**
7. Add a description tag (optional): "Third party upload access"
8. Click **Create access key**
9. **IMPORTANT:** Download the CSV file or copy both keys now. The Secret Access Key will never be shown again.

---

## Step 4: Information to Share

Provide the third party with the following information:

| Item | Value |
|------|-------|
| **Access Key ID** | `AKIA...` (from Step 3) |
| **Secret Access Key** | `...` (from Step 3) |
| **Bucket Name** | `your-bucket-name` |
| **Region** | `us-east-1` (or your bucket's region) |
| **Upload Path** | `uploads/` |

---

## How They Can Upload

### Option A: AWS CLI

```bash
# Configure credentials
aws configure
# Enter: Access Key ID, Secret Access Key, Region, output format (json)

# Upload a file
aws s3 cp myfile.png s3://YOUR-BUCKET-NAME/uploads/myfile.png
```

### Option B: Python (boto3)

```python
import boto3

s3 = boto3.client('s3',
    aws_access_key_id='ACCESS_KEY_ID',
    aws_secret_access_key='SECRET_ACCESS_KEY',
    region_name='us-east-1'
)

# Upload a file
s3.upload_file('local_file.png', 'YOUR-BUCKET-NAME', 'uploads/local_file.png')
```

### Option C: cURL with Pre-signed URL

If you prefer not to share credentials, you can generate pre-signed URLs:

```python
import boto3

s3 = boto3.client('s3',
    aws_access_key_id='YOUR_KEY',
    aws_secret_access_key='YOUR_SECRET',
    region_name='us-east-1'
)

# Generate upload URL valid for 1 hour
url = s3.generate_presigned_url(
    'put_object',
    Params={
        'Bucket': 'YOUR-BUCKET-NAME',
        'Key': 'uploads/filename.png'
    },
    ExpiresIn=3600  # 1 hour
)

print(url)  # Share this URL
```

They can then upload with:
```bash
curl -X PUT -T myfile.png "THE_PRESIGNED_URL"
```

---

## Security Best Practices

1. **Rotate keys regularly** - Create new access keys and disable old ones periodically
2. **Use specific paths** - Restrict uploads to a specific folder, not the entire bucket
3. **Monitor usage** - Enable CloudTrail to log all S3 access
4. **Set bucket policies** - Consider additional bucket-level restrictions
5. **Use pre-signed URLs** - For one-time uploads, pre-signed URLs are more secure than sharing credentials

---

## Revoking Access

To revoke access:

1. Go to **IAM > Users**
2. Click on the upload user
3. Go to **Security credentials** tab
4. Under **Access keys**, click **Actions > Deactivate** or **Delete**

Or delete the entire user:
1. Go to **IAM > Users**
2. Select the user
3. Click **Delete**

---

## Troubleshooting

### "Access Denied" Error

- Verify the bucket name and path in the policy match exactly
- Ensure the policy is attached to the user
- Check that the upload path includes the correct prefix (e.g., `uploads/`)

### "Invalid Access Key" Error

- Double-check the Access Key ID (no extra spaces)
- Verify the key hasn't been deactivated

### "Signature Does Not Match" Error

- Double-check the Secret Access Key
- Ensure the region is correct

