============
Contributing
============

Bug reports, feature suggestions and other contributions are greatly
appreciated! While I can't promise to implement everything, I will always try
to respond in a timely manner.

Short version
=============

* Submit bug reports and feature requests at `GitHub <https://github.com/aburrell/ocbpy/issues>`_
* Make pull requests to the ``develop`` branch

Bug reports
===========

When `reporting a bug <https://github.com/aburrell/ocbpy/issues>`_ please
include:

* Your operating system name and version
* Any details about your local setup that might be helpful in troubleshooting
* Detailed steps to reproduce the bug

Feature requests and feedback
=============================

The best way to send feedback is to file an issue at
`GitHub <https://github.com/aburrell/ocbpy/issues>`_.

If you are proposing a feature:

* Explain in detail how it would work.
* Keep the scope as narrow as possible, to make it easier to implement.
* Remember that this is a volunteer-driven project, and that code contributions
  are welcome :)

Development
===========

To set up `ocbpy` for local development:

1. `Fork ocbpy on GitHub <https://github.com/aburrell/ocbpy/fork>`_.
2. Clone your fork locally::

    git clone git@github.com:your_name_here/ocbpy.git

3. Create a branch for local development::

    git checkout -b name-of-your-bugfix-or-feature

   Now you can make your changes locally. Add tests for bugs and new features
   into the ``ocbpy/tests/`` directory, either in the appropriately named file
   (for changes to an existing file) or in a new file (that should share the
   name of the new file, prepended by ``test_``.  The tests use unittest.
   Changes or additions to the documentation (located in ``docs``) should also
   be made at this time.

4. When you're done making changes, run the tests locally before submitting a
   pull request

5. Commit your changes and push your branch to GitHub::

    git add .
    git commit -m "Brief description of your changes"
    git push origin name-of-your-bugfix-or-feature

6. Submit a pull request through the GitHub website. Pull requests should be
   made to the ``develop`` branch.

Pull Request Guidelines
-----------------------

If you need some code review or feedback while you're developing the code, just
make a pull request.

Do not merge any pull requests, the local maintainer is in charge of merging
until this project grows.

Tips
----

To run a subset of tests from the test directory for a specific environment::

    python test_name.py

To run all the tests for a specific environment::

    python setup.py test
