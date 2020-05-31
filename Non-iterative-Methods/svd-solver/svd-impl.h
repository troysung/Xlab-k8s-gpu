namespace splab
{

	template <typename Real>
	class SVD
	{

	public:

        SVD();
		~SVD();

        void dec( const Matrix<Real> &A );
		Matrix<Real> getU() const;
		Matrix<Real> getV() const;
		Matrix<Real> getSM();
		Vector<Real> getSV() const;

		Real norm2() const;
		Real cond() const;
		int  rank();

    private:

		// the orthogonal matrix and singular value vector
		Matrix<Real> U;
		Matrix<Real> V;
		Vector<Real> S;

        // docomposition for matrix with rows >= columns
		void decomposition( Matrix<Real>&, Matrix<Real>&,
                            Vector<Real>&, Matrix<Real>& );

	};
	// class SVD


    #include <svd-impl.h>

}
// namespace splab


#endif