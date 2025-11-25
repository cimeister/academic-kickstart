#!/bin/bash
set -e

rm -rf public
git submodule add -f -b main https://github.com/cimeister/cimeister.github.io.git public

printf "\033[0;32mDeploying updates to GitHub...\033[0m\n"

# Build the project.
hugo --gc -D -F

# Go To Public folder
cd public

vi .nojekyll
# Add changes to git.
git add -A

# Commit changes.
msg="rebuilding site $(date)"
if [ -n "$*" ]; then
	msg="$*"
fi
git commit -m "$msg"

# Push source and build repos.
git push -f origin main

cd ..
git add -A
git commit -m "$msg"
git push origin main

# SOURCE='.'
# DESTINATION=public/

# TEMP=`mktemp -d`
# echo "Building from $SOURCE"
# hugo --source="$SOURCE" --destination="$TEMP"
# cp "$DESTINATION"/{.git,CNAME} $TEMP
# if [ $? -eq 0 ]; then
#     echo "Syncing to $DESTINATION"
#     rsync -aq --delete "$TEMP/" "$DESTINATION"
# fi
