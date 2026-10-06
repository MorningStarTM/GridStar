# GridStar Data-Generation Infra (Pulumi)

One EC2 instance per episode range, fully automated: Python 3.11 setup, repo
clone, dependency install (CPU-only torch), and both `data_main.py` +
`monitor_server.py` launched unattended on boot. No manual SSH setup needed.

## Files in this folder

| File | What it is |
|---|---|
| `__main__.py` | The actual infrastructure program — defines the security group, AMI lookup, and EC2 instance |
| `Pulumi.yaml` | **Project**-level config (shared by every stack) |
| `Pulumi.<stack-name>.yaml` | **Stack**-level config (one file per instance — episode range, seed, etc.) |
| `requirements.txt` | Python deps for the Pulumi program itself (`pulumi`, `pulumi_aws`) — not GridStar's own deps |
| `venv/` | Local Python virtualenv for running Pulumi — gitignored, never commit it |

## Key concept: Project vs. Stack

- **Project** = this whole `infra/` folder. One project, defined once in `Pulumi.yaml`.
- **Stack** = one deployed *instance* of the project — in our case, one EC2 instance covering one episode range. You can have many stacks from the same project.

Example: `ep-600-800` and `ep-300-500` are two different stacks, each with its own `Pulumi.<name>.yaml` config file, each launching a separate EC2 instance.

---

## `Pulumi.yaml` — project file (you won't usually touch this)

```yaml
name: GridStar
description: data collection from l2rpn grid
runtime:
  name: python
  options:
    toolchain: pip
    virtualenv: venv
config:
  pulumi:tags:
    value:
      pulumi:template: aws-python
```

| Keyword | Meaning |
|---|---|
| `name` | Project name — must match what `__main__.py` and every stack belong to |
| `description` | Free-text, shown in Pulumi Cloud/CLI listings |
| `runtime.name` | Which language the program is written in (`python` here) |
| `runtime.options.toolchain` | Which Python package manager to use (`pip`) |
| `runtime.options.virtualenv` | Folder name for the local venv (`venv`) |
| `config` | Project-wide defaults — rarely edited directly; stack files override this |

---

## `Pulumi.<stack>.yaml` — stack file (one per instance, this is what you edit)

Example, `Pulumi.ep-600-800.yaml`:

```yaml
config:
  aws:region: eu-north-1
  GridStar:startEpisode: "600"
  GridStar:endEpisode: "800"
  GridStar:seed: "7"
  GridStar:keyName: gridstar-key
  GridStar:hfToken:
    secure: AAABAC0qdtMKRbneYL7q/41h6bLGukRI0XaoyX7UEgBVG7uIhh+VEHksI6//8EYbSEJWmD1VBuredEnwftpMM1LqR+09
```

| Keyword | Meaning | Must be unique per stack? |
|---|---|---|
| `aws:region` | Which AWS region to deploy into | No — same everywhere (`eu-north-1`) |
| `GridStar:startEpisode` | First episode (inclusive) this instance generates | **Yes** |
| `GridStar:endEpisode` | Last episode (exclusive) this instance generates | **Yes** |
| `GridStar:seed` | RNG seed — controls random policy choices, chronic shuffling | **Yes** — same seed + overlapping range = duplicate data |
| `GridStar:keyName` | Existing EC2 key pair name, used for SSH | No — reuse the same key pair everywhere |
| `GridStar:hfToken` | HuggingFace write token | No — same HF account for every instance. Shown `secure:` — **encrypted**, safe to commit |

The `GridStar:` prefix matches the project `name` in `Pulumi.yaml` — Pulumi namespaces config keys by project so multiple projects' configs never collide.

**Note on `myIp`:** this is *not* stored in the stack YAML — it's read from an environment variable (`MY_IP`) instead, since your real IP is a minor privacy exposure that's better kept out of git entirely. See below.

---

## Creating a brand-new stack (new episode range) — step by step

**1. Initialize the stack:**
```powershell
cd e:\github_clone\GridStar\infra
pulumi stack init ep-<start>-<end>        # e.g. ep-300-500
```
This both creates `Pulumi.ep-<start>-<end>.yaml` on disk AND registers the stack in Pulumi's backend. If you manually create/copy the YAML file first without running `pulumi stack init`, you'll get `error: no stack named '...' found` — the file alone isn't enough.

**2. Make sure it's the active stack** (config/deploy commands always act on whichever stack is selected):
```powershell
pulumi stack select ep-<start>-<end>
```

**3. Set region:**
```powershell
pulumi config set aws:region eu-north-1
```

**4. Set episode range** (pick a range that doesn't overlap your other running stacks):
```powershell
pulumi config set startEpisode 300
pulumi config set endEpisode 500
```

**5. Set a seed** — must differ from every other stack you've deployed so far:
```powershell
pulumi config set seed 13
```

**6. Set your IP as an environment variable** (NOT via `pulumi config set` — kept out of the YAML/git entirely since it's your real IP address):
```powershell
$env:MY_IP = "<your-current-ip>/32"
```
`__main__.py` reads this directly via `os.environ["MY_IP"]`. This must be set in **every terminal session** before running `pulumi preview`/`pulumi up` — it does not persist across terminal restarts. If you forget, Pulumi fails immediately with a `KeyError: 'MY_IP'` (a safe failure — nothing gets touched on AWS).

Check your current IP hasn't drifted since your last deploy before setting this (ISPs often rotate it).

**7. Set the key pair** (same one reused across all stacks):
```powershell
pulumi config set keyName gridstar-key
```

**8. Set the HF token as a secret** (encrypts it before writing to the YAML file):
```powershell
pulumi config set --secret hfToken hf_xxxxxxxxxxxx
```

**9. Preview before deploying** — always check the plan first:
```powershell
pulumi preview
```
Confirm it shows `+3 to create` and the right episode range/seed baked into the resource names.

**10. Deploy:**
```powershell
pulumi up
```
Confirm `yes` when prompted.

**11. Get connection details:**
```powershell
pulumi stack output
```
Prints `public_ip`, `dashboard_url`, `ssh_command`.

**12. Wait ~3-5 minutes** for the startup script to finish installing Python 3.11, torch, and dependencies, then open `dashboard_url` in a browser to confirm generation started with the right episode range.

---

## Migrating an existing stack off `myIp` config

If a stack already has `GridStar:myIp` set in its config (from before this change), remove it so it stops being written to the YAML file:
```powershell
pulumi stack select ep-<start>-<end>
pulumi config rm myIp
```
Then set `$env:MY_IP` as shown above before your next `pulumi preview`/`pulumi up`. As long as the IP value is the same as before, Pulumi reports no disruptive changes — the Security Group rule stays identical, and the EC2 instance itself is never touched (SG updates don't affect anything already running inside the instance).

**Note:** removing `myIp` from the config only prevents it from being written *going forward*. If an old commit already has your IP in a `Pulumi.<stack>.yaml` file, it still exists in git history — this isn't a credential, so a history rewrite usually isn't worth the hassle, but worth knowing if you care about scrubbing it fully.

## Useful commands reference

| Command | What it does |
|---|---|
| `pulumi stack ls` | List all stacks in this project |
| `pulumi stack select <name>` | Switch the active stack |
| `pulumi config` | Show all config values for the active stack |
| `pulumi config set <key> <value>` | Set a plain config value |
| `pulumi config set --secret <key> <value>` | Set an encrypted (secret) config value |
| `pulumi preview` | Show what would change, without applying anything |
| `pulumi up` | Apply changes — actually creates/updates AWS resources |
| `pulumi stack output` | Show this stack's exported outputs (IP, URLs, etc.) |
| `pulumi destroy` | Tear down all resources for the active stack |
| `pulumi stack rm <name>` | Remove a stack's config entirely (after `destroy`) |

## Tearing down an instance when done

```powershell
pulumi stack select ep-<start>-<end>
pulumi destroy        # removes the EC2 instance + security group
pulumi stack rm ep-<start>-<end>   # optional — removes the stack's config file/registration too
```
