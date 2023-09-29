Contributing to opendataval
===========================

There are many ways to contribute, including documentation, [issues](https://github.com/opendataval/opendataval/issues), or completeting a [challenge](https://opendataval.github.io/leaderboards.html) for the leaderboard! To contribute to the source code, we're always in the need of more datasets, data evaluators, models, or experiments. Documentation is found in the [docs/](https://github.com/opendataval/opendataval/tree/main/docs/) directory.

Another way to contribute is to report issues you're facing, and give a "thumbs up" on issues that others reported and that are relevant to you. It also helps us if you spread the word: reference the project from your blog and articles, link to it from your website, or simply star it in GitHub to say "I use it".

Contributing
------------
1. [Create an account on GitHub](https://github.com/) if you do not already have one.
2. ```git clone git@github.com:opendataval/opendataval.git```
3. ```make install-dev```
4. Create a feature branch ```git checkout -b my_feature```
5. Update the branch frequently with ```git fetch origin```
6. Develop the feature. Feel free to use ```git commit --amend``` so each commit feels like a checkpoint with detailed messages. With precommit hooks, you're forced to adhere to some style guidelines.
7. Test your code (and write more [tests](https://github.com/opendataval/opendataval/tree/main/test)!) using ```make coverage```
8. Keep your code clean with ```make format```
9. Send it to a remote branch with ```git push -u origin ```

Pull Request Checklist
----------------------
1. Give your pull request a helpful title that summarizes what your contribution does. In some cases “Fix <ISSUE TITLE>” is enough.
2. Make sure all test cases past and good documentation/comments is added for new features. We use the numpy style documentation.
3. If merging your pull request means that some other issues/PRs should be closed, you should [use keywords to create link to them](https://github.blog/2013-05-14-closing-issues-via-pull-requests/) (e.g., `Fixes #1234`; multiple issues/PRs are allowed as long as each one is preceded by a keyword).
4. Please stick around! New features may require maintenance so there's always more to contribute.


Quick links
-----------
* [Paper](https://arxiv.org/abs/2306.10577)
* [Issues](https://github.com/opendataval/opendataval/issues)
* [Scikit-learn docs](https://scikit-learn.org/dev/developers/contributing.html)

Code of Conduct
---------------
We abide by the principles of openness, respect, and consideration of others
of the Python Software Foundation: https://www.python.org/psf/codeofconduct/.