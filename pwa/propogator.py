{% macro Prop()%}
    def BW(self, m_,w_,Sbc):
        l = (Sbc.shape)[0]
        temp = dplex.dconstruct(m_*m_ - Sbc,  -m_*w_*np.ones(l))
        return dplex.ddivide(1.0, temp)

    def BW_relativity(self, m_,w_,Sbc):
        gamma=np.sqrt(m_*m_*(m_*m_+w_*w_))
        k = np.sqrt(2*np.sqrt(2)*m_*np.abs(w_)*gamma/np.pi/np.sqrt(m_*m_+gamma))
        l = (Sbc.shape)[0]
        temp = dplex.dconstruct(m_*m_ - Sbc,  -m_*w_*np.ones(l))
        return dplex.ddivide(k, temp)

    def flatte980(self,m_,g_pipi,rg,Sbc):
        g_kk = rg * g_pipi
        m_k = 0.493677
        m_pi = 0.13957061
        rho_kk = np.sqrt(np.abs(1 - 4*m_k*m_k / Sbc))
        rho_pipi = np.sqrt(np.abs(1 - 4*m_pi*m_pi / Sbc))
        tmp_A = dplex.dconstruct(m_**2 - Sbc, -1*(g_pipi*rho_pipi + g_kk*rho_kk))
        return dplex.ddivide(1.0, tmp_A)

    def flatte1270(self,m_,w_,Sbc):
        rm = m_ * m_
        gr = m_ * w_
        q2r = 0.25 * rm - 0.0194792
        b2r = q2r * (q2r + 0.1825) + 0.033306
        g11270 = gr * b2r / np.power(q2r,2.5)
        q2 = 0.25 * Sbc - 0.0194792
        b2 = q2 * (q2 + 0.1825) + 0.033306
        g1 = g11270 * np.power(q2,2.5) / b2
        tmp = dplex.dconstruct(Sbc - rm, g1)
        return dplex.ddivide(gr, tmp)
    
    def flatte500(self,m_,b1,b2,b3,b4,b5,Sbc):
        m2 = m_*m_
        rp = 0.139556995
        mpi2d2 = 0.009739946882
        cro1 = np.sqrt(np.abs((Sbc-(2*rp)**2)*Sbc))/Sbc
        cro2 = np.sqrt(np.abs((m2-(2*rp)**2)*m2))/m2
        pip1 = np.sqrt(np.abs(1.0 - 0.3116765584/Sbc))/(1.0+np.exp(9.8-3.5*Sbc))
        pip2 = np.sqrt(np.abs(1.0 - 0.3116765584/m2))/(1.0+np.exp(9.8-3.5*m2)) # ?
        cgam1 = m_*(b1+b2*Sbc)*(Sbc-mpi2d2)/(m2-mpi2d2)*np.exp(-(Sbc-m2)/b3)*cro1/cro2
        cgam2 = m_*b4*pip1/pip2
        tmp = dplex.dconstruct(m2-Sbc, -b5*(cgam1+cgam2))
        return dplex.ddivide(1.0, tmp)
{% endmacro %}