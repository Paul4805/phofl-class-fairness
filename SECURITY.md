# Security Policy

## Supported versions

Only the current `main` branch is supported.

## Reporting a vulnerability

Please do **not** open a public issue for security-sensitive reports. Use
GitHub’s private vulnerability reporting:

https://github.com/Paul4805/phofl-class-fairness/security/advisories/new

Include:

- a description of the issue;
- the affected file or workflow;
- a minimal reproduction, if possible;
- any suggested remediation.

You should receive an acknowledgement within a few days. Please allow time for
a fix before public disclosure.

## Scope

This repository contains research code and result artifacts. It does not ship
credentials, model checkpoints, or datasets. The main security concerns are:

- accidental credential or private-data commits;
- unsafe deserialization when loading third-party checkpoints (`torch.load`);
- untrusted notebook execution.

The recommended environment is an isolated virtual environment on a machine
you control. Do not run untrusted notebooks or load untrusted `.pt` files.
