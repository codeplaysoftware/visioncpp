set -e

mkdir docs
cd docs
git clone -b gh-pages https://git@$GH_REPO ./

git config --global push.default simple
git config user.name "Travis CI"
git config user.email "travis@travis-ci.org"

# remove all
rm -rf *

if [ -d "../build/html" ] && [ -f "../build/html/visioncpp_8hpp.html" ]; then
  # we build docs already, so we can move it
  mv ../build/html/* ./
  git add --all

  git commit -m "Deploy code docs to GitHub Pages Travis build: ${TRAVIS_BUILD_NUMBER}" -m "Commit: ${TRAVIS_COMMIT}"
  git push --force "https://${GH_REPO_TOKEN}@${GH_REPO}" > /dev/null 2>&1
else
  echo '' >&2
  echo 'Warning: No documentation has been found.' >&2
  exit 1
fi
