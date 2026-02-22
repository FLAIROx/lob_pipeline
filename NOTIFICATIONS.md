# Notifications Setup

The pipeline supports optional ntfy.sh notifications for job completion alerts.

## Quick Setup

### Option 1: Environment Variables (Recommended)
```bash
export NTFY_TOPIC_INFERENCE="my_inference_alerts"
export NTFY_TOPIC_BENCHMARKS="my_benchmark_alerts"
./pipeline/run_lobbench_pipeline.sh <CHECKPOINT_PATH>
```

### Option 2: ~/.ntfy-topic File
```bash
echo "my_alerts_topic" > ~/.ntfy-topic
./pipeline/run_lobbench_pipeline.sh <CHECKPOINT_PATH>
```
This will use the same topic for both inference and benchmark notifications.

### Option 3: Edit config.sh (Not Recommended for Shared Repos)
Edit `pipeline/config.sh` to hardcode your topics:
```bash
NTFY_TOPIC_INFERENCE="my_inference_alerts"
NTFY_TOPIC_BENCHMARKS="my_benchmark_alerts"
```

## How It Works

- **Disabled by default**: If no topic is configured, no notifications are sent
- **Per-user**: Each user sets their own topics (no shared notifications)
- **Requires slurm-notify**: Uses `~/bin/slurm-notify` to send messages

## Notification Messages

### Inference Jobs
- **Success**: `infer_<name>_<stock> (<job_id>) finished (exit=0)`
- **Failure**: `infer_<name>_<stock> (<job_id>) finished (exit=<code>)`

### Benchmark Jobs
- **Success**: `bench_<name>_<stock> (<job_id>) finished (errors=0)`
- **Failure**: `bench_<name>_<stock> (<job_id>) FAILED: <reason>`

## Setting Up ntfy.sh

1. Install slurm-notify (if not already installed):
   ```bash
   # See your team's documentation for slurm-notify setup
   ```

2. Create your ntfy.sh topic:
   - Go to https://ntfy.sh
   - Choose a unique topic name (e.g., `<name>_lob_alerts_xk2p9`)
   - Subscribe to it on your phone or browser

3. Configure the pipeline to use your topic (see options above)

## Privacy Note

- **Never commit your ~/.ntfy-topic to git**
- **Use unique, hard-to-guess topic names** (ntfy.sh topics are public)
- Example good topic: `<name>_lob_pipeline_dk39x2m`
- Example bad topic: `lob_alerts` (too common, others might guess it)

## Disabling Notifications

To disable notifications, simply don't set any topic:
```bash
unset NTFY_TOPIC_INFERENCE
unset NTFY_TOPIC_BENCHMARKS
./pipeline/run_lobbench_pipeline.sh <CHECKPOINT_PATH>
```

## Example Workflow

```bash
# Set up once (add to ~/.bashrc for persistence)
export NTFY_TOPIC_INFERENCE="<name>_inference_k3x9"
export NTFY_TOPIC_BENCHMARKS="<name>_benchmarks_k3x9"

# Run pipeline - you'll get notified when jobs complete
cd /lus/lfs1aip2/projects/s5e/lob_pipeline
./pipeline/run_lobbench_pipeline.sh \
    /lus/lfs1aip2/projects/s5e/quant/AlphaTrade/LOBS5/checkpoints/logical-serenity-19_4dhsl6me


```
