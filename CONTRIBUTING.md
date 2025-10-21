# Contributing to Smart Stays

First off, thank you for considering contributing to Smart Stays! It's people like you that make Smart Stays such a great tool.

## Where do I go from here?

If you've noticed a bug or have a feature request, [make one](https://github.com/your-username/smart-stays/issues/new)! It's generally best if you get confirmation of your bug or approval for your feature request this way before starting to code.

## Fork & create a branch

If this is something you think you can fix, then [fork Smart Stays](https://github.com/your-username/smart-stays/fork) and create a branch with a descriptive name.

A good branch name would be (where issue #325 is the ticket you're working on):

```sh
git checkout -b 325-add-japanese-translations
```

## Get the test suite running

Make sure you're running the test suite locally before you start making changes.

## Implement your fix or feature

At this point, you're ready to make your changes! Feel free to ask for help; everyone is a beginner at first :smile_cat:

## Make a Pull Request

At this point, you should switch back to your master branch and make sure it's up to date with the latest upstream version of the code.

```sh
git remote add upstream git@github.com:your-username/smart-stays.git
git checkout master
git pull upstream master
```

Then update your feature branch from your local copy of master, and push it!

```sh
git checkout 325-add-japanese-translations
git rebase master
git push --set-upstream origin 325-add-japanese-translations
```

Finally, go to GitHub and [make a Pull Request](https://github.com/your-username/smart-stays/compare/master...325-add-japanese-translations)

## Keeping your Pull Request updated

If a maintainer asks you to "rebase" your PR, they're saying that a lot of code has changed, and that you need to update your branch so it's easier to merge.

To learn more about rebasing and merging, check out this guide on [understanding the GitHub flow](https://guides.github.com/introduction/flow/).

## Merging a PR (for maintainers)

Thank you for your contribution!
