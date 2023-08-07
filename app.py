import streamlit as st
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib import ticker
import os
import locale
locale.setlocale( locale.LC_ALL, '' )

import mymodule

#arr = mymodule.make_data()

# 1 made empty repository on github
# 2 PyCharm Project from github: .py script that is github, made script & requirements.txt, commit & pushed
# 3 Log into streamlit, and app is there...

# PRIORS - > USER INPUT
st.header('Should you enter the geothermal lottery without further information?')
st.subheader('What\'s the Prior Probability $Pr(.)$ of a POSITIVE geothermal site?')
Pr_prior_POS_demo = mymodule.Prior_probability_binary() #np.linspace(0.05,1,20) 

#### start of paste  -> CHANGE to input
count_ij = np.zeros((2,10))
value_array = mymodule.make_value_array(count_ij, profit_drill_pos= 1e6, cost_drill_neg = -1e6)
#st.write('value_array', value_array)

## Calculate Vprior
#f_VPRIOR(X_unif_prior, value_array, value_drill_DRYHOLE[-1])  
value_drill_DRYHOLE = np.linspace(100, -1e6,10)

# NOT TRUE: This function can be called with multiple values of "dry hole"
# vprior_INPUT_demo = mymodule.f_VPRIOR([1-Pr_prior_POS_demo[0],Pr_prior_POS_demo[0]], value_array, value_drill_DRYHOLE[-1])       
vprior_INPUT_min = mymodule.f_VPRIOR([0.9,0.1], value_array, value_drill_DRYHOLE[-1])  
vprior_INPUT_max = mymodule.f_VPRIOR([0.9,0.1], value_array, value_drill_DRYHOLE[0])   
VPI_max = mymodule.Vperfect(Pr_prior_POS_demo, value_array,  value_drill_DRYHOLE[0])   
# l2 = list(map(lambda v: v ** 2, l1))
vprior_INPUT_demo_list = list(map(lambda vv: mymodule.f_VPRIOR([1-Pr_prior_POS_demo,Pr_prior_POS_demo], 
                                                              value_array,vv),value_drill_DRYHOLE))
st.subheader('Yes if Vprior is positive. Vprior with $Pr(POSITIVE)$='+str(Pr_prior_POS_demo))  #Pr_prior_POS_demo[0]
st.write(r'''$V_{prior} =  \max\limits_a \Sigma_{i=1}^2 Pr(X = x_i)  v_a(x_i) \ \  \forall a $''')


showVperfect = st.checkbox('Show Vperfect')

firstfig, ax = plt.subplots()
plt.plot(value_drill_DRYHOLE, vprior_INPUT_demo_list,'g.-', linewidth=5,label='$V_{prior}$')
plt.ylabel('$V_{prior}$',fontsize=14)
plt.xlabel('Dryhole Cost')
if showVperfect:  
    VPIlist = list(map(lambda uu: mymodule.Vperfect(Pr_prior_POS_demo, value_array,uu),value_drill_DRYHOLE))
    # st.write('VPI',np.array(VPIlist),vprior_INPUT_demo_list)
    VOIperfect = np.maximum((np.array(VPIlist)-np.array(vprior_INPUT_demo_list)),np.zeros(len(vprior_INPUT_demo_list)))
    # VPI_list = list(map(lambda v: mymodule.f_Vperfect(Pr_prior_POS_demo, value_array, v), value_drill_DRYHOLE))
    plt.plot(value_drill_DRYHOLE,VPIlist,'b', label='$V_{perfect}$')
    plt.plot(value_drill_DRYHOLE,VOIperfect,'b--', label='$VOI_{perfect}$')
plt.legend(loc=3)
plt.ylim([vprior_INPUT_min,(VPI_max+20)]) #-vprior_INPUT_min
# additional code before plt.show()
formatter = ticker.ScalarFormatter()
formatter.set_scientific(False)
ax.yaxis.set_major_formatter(formatter)
ax.xaxis.set_major_formatter(formatter)
ax.xaxis.set_major_formatter('${x:1.0f}')
st.pyplot(firstfig)

if showVperfect:  
    st.write(r'''$VOI_{perfect}$ = V_{perfect}-V_{prior}='''+str(VPIlist[0])+' - '+str(vprior_INPUT_demo_list[0]))
    st.write('Since you "know" when either subsurface condition occurs, you can pick the best ($\max\limits_a$) drilling altervative first ($v_a$).')
    st.write(r'''$V_{perfect} =  \Sigma_{i=1}^2 Pr(X = x_i) \max\limits_a v_a(x_i) \ \  \forall a $''')


with st.sidebar:
            
    # LOCATION OF THIS FILE (Carbonate Aquifer only to start?)
    uploaded_files = st.file_uploader("Choose a POS :fire: & NEG :thumbsdown: file",type=['csv'],accept_multiple_files=True)
    st.write(len(uploaded_files))

    if uploaded_files is not None and len(uploaded_files)==2:
        for uploaded_file in uploaded_files:
            # bytes_data = uploaded_file.read()
            st.write("filename:", uploaded_file.name)
            if uploaded_file.name[0:3]=='NEG':
                neg_upload_file = uploaded_file
                dfN = pd.read_csv(neg_upload_file)
                st.write('NEG File preview...')
                st.write(dfN.head())
            elif uploaded_file.name[0:3]=='POS':
                pos_upload_file = uploaded_file
                df = pd.read_csv(pos_upload_file)
            else:
                st.write('Dude, you didn\'t select a POS and NEG file, try again')

        if pos_upload_file.name[3:7] != neg_upload_file.name[3:7]:
                st.write('You aren\'t comparing data from the same region. STOP!')
        else:        
            st.subheader('ML Nevada Data')
            st.subheader('Choose attribute for VOI calculation')
            attribute0 = st.selectbox('What attributes would you like to calculate', df.columns) 
    # attribute0 = st.multiselect('What attributes would you like to calculate', df.columns,max_selections=2)
    
    # with st.echo(): # this prints out 
    # st.write("Update dry hole loss in sidebar.")
    # number = st.number_input(r'''$v_{a=drill}(\theta=Negative)$''')
    # st.write('The current number is ', number)

    #with st.spinner("Loading..."):
    #    time.sleep(5)
    #st.success("Done!")

st.title('Domain Likelihoods: thresholding distances to labels')

# uploaded_fileNEG = st.file_uploader("Choose a NEG file",type=['csv'])

if uploaded_file is not None:
    # df = pd.read_csv(uploaded_file)
    # dfN = pd.read_csv(file_path+neg_upload_file)
    # st.subheader('ML Nevada Data')
    # st.write('File preview...')
    # st.write(df.head())

    # st.write(df)
      
    if attribute0 is not None:
        st.write('You picked this attribute ', attribute0)

        x_cur = attribute0
    
        screen_att0 ='PosSite_Di'
        screen_att1 ='NegSite_Di'
        y_cur0 = 'GeodeticStrainRate'  # hard code for now will come from multiselect
        # xmax_cur

        df_screen = df[df[x_cur]>-9999]
        df_screenN = dfN[dfN[x_cur]>-9999]
        st.write('dataframe is shape: {thesize}'.format(thesize=df_screen.shape))
        st.write('attribute stats ', df_screen[attribute0].describe())

        distance_meters = st.slider('Change likelihood by *screening* distance to positive label [km or meters??]',10, int(np.max(df_screen['PosSite_Distance'])-10), 800, step=100) # min, max, default
        # NEG_distance_meters = st.slider('Change likelihood by *screening* distance to negative label [km or meters??]', 
        #     10, int(np.max(df_screenN['NegSite_Di'])-10), int(np.median(df_screenN['NegSite_Di'])), step=1000)

        # round to make sure it rounds to nearest 10
        dfpair0 = df_screen[(df_screen['PosSite_Distance'] <=round(distance_meters,-1))] 
        print('dfpair0.head(10)', dfpair0.head(10))
        dfpair = dfpair0[dfpair0[y_cur0]>-9999] 
        # # # OJO : may want to keep this off until have it for NEG 
        dfpairN = df_screenN#[(df_screenN['NegSite_Di'] <=round(NEG_distance_meters,-1))] 
        
        st.subheader('Calculate & Display Likelihoods')
        st.write('We can compute this "empirical" likelihood with the counts of interpretations.')
        mymodule.my_kdeplot(dfpair,x_cur,y_cur0,dfpairN)
        #waiting_condition = 1
        #while (waiting_condition):
        #    st.image('https://media.giphy.com/media/gu9XBXiz60HlO5p9Nz/giphy.gif')

        # waiting_condition = mymodule.my_kdeplot(dfpair,x_cur,y_cur0,y_cur1,waiting_condition)
        
        # split up if we want to test bandwidth 
        X_train, X_test, y_train, y_test = mymodule.make_train_test(dfpair,x_cur,y_cur0,dfpairN)
 
        best_params = mymodule.optimal_bin(X_train, y_train)

        # Likelihood via KDE estimate
        predictedLikelihood_pos, predictedLikelihood_neg, x_sampled, count_ij= mymodule.likelihood_KDE(X_train,X_test, y_train, y_test,x_cur,y_cur0, best_params)

      
        #Basic question: How far apart (different) are two distributions P and Q? Measured through distance & divergences
        #https://nobel.web.unc.edu/wp-content/uploads/sites/13591/2020/11/Distance-Divergence.pdf


        st.subheader('Change the Prior :blue['+r'''$Pr(\Theta = \theta_i)$'''+'] of POSITIVE Geothermal site')
        Pr_prior_POS = mymodule.Prior_probability_binary('Prior used in Posterior')

        st.subheader('Posterior ~ :blue[Prior] * Likelhood')
        st.write('*Given that we know the TRUE GEOTHERMAL OUTCOME (remember "$|$" stands for "given"), what is the likelihood of the label GIVEN the data (X) ')
        st.subheader(' :violet['+r'''$Pr(\Theta = \theta_i | X =x_j)$'''+'] ~\
                     :blue['+r'''$Pr(\Theta = \theta_i )$'''+'] \
                     :orange['+r'''$Pr( X=x_j | \Theta = \theta_i )$'''+']')
        st.latex(r''' Pr( \Theta = \theta_i | X =x_j ) = 
        \frac{Pr(\Theta = \theta_i ) Pr( X=x_j | \Theta = \theta_i )}{Pr (X=x_j)} 
        ''')  
        
        # POSTERIOR from WGC
        # posterior = mymodule.Posterior_WGC()
        # POSTERIOR via_Naive_Bayes: Draw back here is ??? can I get the marginal out? 
        post_input, post_uniform = mymodule.Posterior_via_NaiveBayes(Pr_prior_POS,X_train, X_test, y_train, y_test, x_sampled, x_cur)

        value_array = mymodule.make_value_array(count_ij, profit_drill_pos= 1e6, cost_drill_neg = -1e6)
        st.write('value_array', value_array)

        #f_VPRIOR(X_unif_prior, value_array, value_drill_DRYHOLE[-1])  
        value_drill_DRYHOLE = np.linspace(100, -1e6,10)

        # This function can be called with multiple values of "dry hole"
        vprior_unif_out = mymodule.f_VPRIOR([1-Pr_prior_POS,Pr_prior_POS], value_array) #, value_drill_DRYHOLE[-1]       
        st.subheader('Should you enter the geothermal lottery?')
        st.subheader(r'''$V_{prior}$ '''+str(locale.currency(vprior_unif_out, grouping=True )))

        VPI = mymodule.Vperfect(Pr_prior_POS, value_array)
        st.subheader(r'''$VOI_{perfect}$ ='''+str(locale.currency(VPI, grouping=True )))

        # Need a marginal estimate 
        # Calculate marg_input, marg_unif       
        # Passing unscale likelihood?
        Pr_Marg = mymodule.marginal(Pr_prior_POS, predictedLikelihood_pos, predictedLikelihood_neg)
        # st.write(np.shape(Pr_Marg),Pr_Marg[0,-20:],Pr_Marg[1,-20:])
        
        mymodule.Posterior_Marginal_plot(post_input, post_uniform, np.sum(Pr_Marg,0), x_cur, x_sampled)

        # VII_unif = mymodule.f_VIMPERFECT(post_uniform, value_array,Pr_UnifMarg)
        VII_input, VII_unifMarginal= mymodule.f_VIMPERFECT(post_input, value_array, Pr_Marg, x_sampled)
        
        st.subheader(r'''$V_{imperfect}$='''+str(locale.currency(VII_input, grouping=True )))
        st.write('with uniform marginal', locale.currency(VII_unifMarginal, grouping=True ))
     

        