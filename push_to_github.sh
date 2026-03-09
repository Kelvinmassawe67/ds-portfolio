#!/bin/bash
# =====================================================
# GitHub Push Script for DS Portfolio
# =====================================================
# Usage:
#   1. Create a new repo on GitHub (e.g. "ds-portfolio")
#   2. Run: bash push_to_github.sh <github-username> <repo-name>
#      Example: bash push_to_github.sh johndoe ds-portfolio
#
# You'll be prompted for your GitHub personal access token.
# =====================================================

USERNAME=$1
REPO=$2

if [ -z "$USERNAME" ] || [ -z "$REPO" ]; then
  echo "Usage: bash push_to_github.sh <github-username> <repo-name>"
  exit 1
fi

echo "Pushing to https://github.com/$USERNAME/$REPO"
echo ""
echo "Enter your GitHub Personal Access Token (hidden):"
read -s TOKEN

git remote remove origin 2>/dev/null
git remote add origin https://$USERNAME:$TOKEN@github.com/$USERNAME/$REPO.git
git branch -M main
git push -u origin main

echo ""
echo "✓ Portfolio pushed to: https://github.com/$USERNAME/$REPO"
