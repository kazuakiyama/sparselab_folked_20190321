# INSTALLDIR
all:
	cd src/lbfgsb; make install
	cd src/slsqp; make install
	cd src/finufft; make lib
	cd src/fortlib; make install
#	cd src/mfista; make install

install: all
	python setup.py install --record setup_files.log

clean:
	cd src/lbfgsb; make clean
	cd src/slsqp; make clean
	cd src/finufft; make clean
	cd src/fortlib; make clean
#	cd src/mfista; make clean
	rm -f sparselab/*.pyf
	rm -f sparselab/*.pyc
	rm -rf sparselab/*.so*

uninstall: clean
	cd src/lbfgsb; make uninstall
	cd src/slsqp; make uninstall
	cd src/fortlib; make uninstall
#	cd src/mfista; make uninstall
	rm -rf autom4te.cache
	rm -f config.log
	rm -f config.status
	rm -f configure
	rm makefile
	cat setup_files.log|grep sparselab|xargs rm -rf
	rm -rf setup_files.log *egg* *build* *dist*
