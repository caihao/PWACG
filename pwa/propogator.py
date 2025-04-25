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

    def BW_phi(self,m_,w_,Sbc):
        l = (Sbc.shape)[0]
        m_k = 0.493677
        q_0 = 1/3
        q_m = np.sqrt(Sbc-4*m_k*m_k)/2
        q_phi = np.sqrt(m_*m_-4*m_k*m_k)/2
        b_qm = 1 / np.sqrt(1+(q_m/q_0)*(q_m/q_0))
        b_qphi = 1 / np.sqrt(1+(q_phi/q_0)*(q_phi/q_0))

        k = q_m / q_0 *b_qm
        Gamma_phi = w_ * m_ / np.sqrt(Sbc)*(q_m/q_phi)*(q_m/q_phi)*b_qm*b_qm/b_qphi/b_qphi

        temp = dplex.dconstruct(m_*m_ - Sbc, -m_*Gamma_phi*np.ones(l))
        return dplex.ddivide(k, temp)

    def flatte980(self,m_,g_pipi,rg,Sbc):
        # 对于phipipi的过程的rho_kk的计算，不能单纯取绝对值，因为有一部分pipi在kk的质量阈值之下
        g_kk = rg * g_pipi
        m_k = 0.493677
        m_pi = 0.13957061

        tmp_kk = 1 - 4*m_k*m_k / Sbc
        sign_kk = np.sign(tmp_kk)
        tmp_pipi = 1 - 4*m_pi*m_pi / Sbc
        sign_pipi = np.sign(tmp_pipi)

        # 若这一项大于零，应该留在虚部，如果小于零，应该*加*到实部

        real_kk_factor = ( tmp_kk - sign_kk * tmp_kk ) / (tmp_kk * 2)
        img_kk_factor = ( tmp_kk + sign_kk * tmp_kk ) / (tmp_kk * 2)

        rho_kk = np.sqrt(np.abs(1 - 4*m_k*m_k / Sbc))
        rho_pipi = np.sqrt(np.abs(1 - 4*m_pi*m_pi / Sbc))
        tmp_A = dplex.dconstruct(m_**2 - Sbc + g_kk*rho_kk*real_kk_factor, -1*m_*(g_pipi*rho_pipi + g_kk*rho_kk*img_kk_factor))
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
    
    # def flatte500(self,m_,b1,b2,b3,b4,b5,Sbc):
    #     m2 = m_*m_
    #     rp = 0.139556995
    #     mpi2d2 = 0.009739946882
    #     cro1 = np.sqrt(np.abs((Sbc-(2*rp)**2)*Sbc))/Sbc
    #     cro2 = np.sqrt(np.abs((m2-(2*rp)**2)*m2))/m2
    #     pip1 = np.sqrt(np.abs(1.0 - 0.3116765584/Sbc))/(1.0+np.exp(9.8-3.5*Sbc))
    #     pip2 = np.sqrt(np.abs(1.0 - 0.3116765584/m2))/(1.0+np.exp(9.8-3.5*m2)) # ?
    #     cgam1 = m_*(b1+b2*Sbc)*(Sbc-mpi2d2)/(m2-mpi2d2)*np.exp(-(Sbc-m2)/b3)*cro1/cro2
    #     cgam2 = m_*b4*pip1/pip2
    #     tmp = dplex.dconstruct(m2-Sbc, -b5*(cgam1+cgam2))
    #     return dplex.ddivide(1.0, tmp)

    def flatte500(self,m_,b1,b2,A,Sbc):
        m_pi = 0.139556995
        m_k = 0.493677
        m_eta = 0.547862
        S_a = 0.41*m_pi*m_pi
        m2 = m_*m_
        alpha = 1.3
        g4 = 0.011 # 参考文献里有很多取值

        g1_sqr = m_*(b1+b2*Sbc)*np.exp(-(Sbc-m_*m_)/A)
        rho_pipi = np.sqrt(np.abs(1 - 4*m_pi*m_pi / Sbc))
        rho_M = np.sqrt(np.abs(1 - 4*m_pi*m_pi / (m_*m_)))

        tmp_kk = 1 - 4*m_k*m_k / Sbc
        rho_kk = np.sqrt(np.abs(1 - 4*m_k*m_k / Sbc))
        tmp_nn = 1 - 4*m_eta*m_eta / Sbc
        rho_nn = np.sqrt(np.abs(1 - 4*m_eta*m_eta / Sbc))

        rho_kk_sign = np.sign(1 - 4*m_k*m_k / Sbc) 
        rho_nn_sign = np.sign(1 - 4*m_eta*m_eta / Sbc)

        kk_real_factor = ( tmp_kk - rho_kk_sign * tmp_kk ) / (tmp_kk * 2)
        kk_img_factor = ( tmp_kk + rho_kk_sign * tmp_kk ) / (tmp_kk * 2)
        nn_real_factor = ( tmp_nn - rho_nn_sign * tmp_nn ) / (tmp_nn * 2)
        nn_img_factor = ( tmp_nn + rho_nn_sign * tmp_nn ) / (tmp_nn * 2)
        # pipi 不变质量会出现在 kk 和 etaeta 质量阈值 之下
        
        rho_4pis = 1.0/(1+np.exp(7.082-2.845*Sbc))
        rho_4pim = 1.0/(1+np.exp(7.082-2.845*(m_*m_)))

        mgam1 = g1_sqr*(Sbc-S_a)/(m_*m_-S_a)*rho_pipi
        mgam2 = 0.6*g1_sqr*(Sbc/m2)*np.exp(-1*alpha*np.abs(Sbc-4*m_k*m_k))*rho_kk 
        mgam3 = 0.2*g1_sqr*(Sbc/m2)*np.exp(-1*alpha*np.abs(Sbc-4*m_eta*m_eta))*rho_nn 
        mgam4 = m_*g4*rho_4pis/rho_4pim

        j1_s = (2 + rho_pipi * np.log((1-rho_pipi)/(1+rho_pipi))) / np.pi #log本身就是以e为底
        j1_m = (2 + rho_M * np.log((1-rho_M )/(1+rho_M ))) / np.pi
        zs = j1_s - j1_m 

        # mgam = mgam1 + mgam2 + mgam3 + mgam4 
        T_pipi = dplex.dconstruct(m_*m_ - Sbc - g1_sqr*(Sbc-S_a)/(m_*m_-S_a)*zs + kk_real_factor*mgam2 + nn_real_factor*mgam3 ,mgam1+mgam4 + kk_img_factor*mgam2 +nn_img_factor *mgam3 )

        return dplex.ddivide(1.0,T_pipi)

    def BW_long(self, m_,w_,Sbc):
        # 按照LHCb note 加入BW为1的无共振贡献
        l = (Sbc.shape)[0]
        temp = dplex.dconstruct(m_*m_ - m_*m_ + np.ones(l),  -m_*w_*np.zeros(l))
        return dplex.ddivide(1.0, temp)

{% endmacro %}