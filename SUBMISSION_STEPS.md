# Submission Steps

This checklist starts from building the Docker image and ends with submitting the image URI in the official portal.

Run all commands from the repository root:

```bash
cd /home/user/aic
```

## 1. Build the Docker Image

```bash
docker compose -f docker/docker-compose.yaml build 
```

The compose file builds the `model` service image as:

```text
my-solution:v1
```

## 2. Verify Locally

```bash
docker compose -f docker/docker-compose.yaml up
```

Wait for the local evaluation to finish. Do not submit until the container starts successfully and the `aic_model` lifecycle validation passes.

To stop and clean up the running compose stack:

```bash
docker compose -f docker/docker-compose.yaml down
```

## 3. Configure AWS Credentials

Replace `<team_name>` with the team profile/slug from your onboarding email.

```bash
aws configure --profile intrinsicchallengers
export AWS_PROFILE=intrinsicchallengers
```

Team Slug: intrinsicchallengers
Access Key ID: AKIA6FQQQTC3WECHXGPH
Secret Access Key: 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
Region: us-east-1

Use:

```text
Region: us-east-1
Output format: json
```

## 4. Log In to ECR

```bash
aws ecr get-login-password --region us-east-1 \
  | docker login --username AWS --password-stdin 973918476471.dkr.ecr.us-east-1.amazonaws.com
```

## 5. Tag the Image

Replace `<team_name>` with your assigned team repository name.
Replace `v1` with a fresh tag for every submission, such as `v2`, `v3`, or a Git commit SHA.

```bash
docker tag my-solution:v1 973918476471.dkr.ecr.us-east-1.amazonaws.com/aic-team/intrinsicchallengers:v1
```

## 6. Push the Image

```bash
docker push 973918476471.dkr.ecr.us-east-1.amazonaws.com/aic-team/intrinsicchallengers:v1
```

Important: ECR tags are immutable. If you already pushed `:v1`, use a new tag such as `:v2`.

## 7. Submit in the Portal

Copy the full pushed image URI:

```text
973918476471.dkr.ecr.us-east-1.amazonaws.com/aic-team/<team_name>:v1
```

Then:

1. Open the official submission portal.
2. Select `AI for Industry Challenge`.
3. Go to `Submit`.
4. Select the `Qualification` phase.
5. Paste the full image URI into the `OCI Image` field.
6. Submit.

## Useful Checks

Check Docker disk usage:

```bash
docker system df
```

Check host disk space:

```bash
df -h
```

Inspect recent model logs during local verification:

```bash
docker compose -f docker/docker-compose.yaml logs --tail=300 model
```

Inspect recent eval logs:

```bash
docker compose -f docker/docker-compose.yaml logs --tail=300 eval
```
