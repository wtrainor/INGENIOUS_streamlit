import streamlit as st
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib import ticker
import os
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
# import babel.numbers
# import decimal
#import locale
#locale.setlocale( locale.LC_ALL, '' )

import mymodule

#arr = mymodule.make_data()

# 1 made empty repository on github
# 2 PyCharm Project from github: .py script that is github, made script & requirements.txt, commit & pushed
# 3 Log into streamlit, and app is there...

# PRIORS - > USER INPUT
st.header('Probability of geothermal success given information available today?')
#st.write('What\'s the Prior Probability of a POSITIVE geothermal site?  $Pr(x=Positive)$')
Pr_prior_POS_demo = mymodule.Prior_probability_binary() 

#### start of paste  -> CHANGE to input
count_ij = np.zeros((2,10))
value_array, value_array_df = mymodule.make_value_array(count_ij, profit_drill_pos= 1e6, cost_drill_neg = -1e6)
# # st.write('value_array', value_array)

## Calculate Vprior
#f_VPRIOR(X_unif_prior, value_array, value_drill_DRYHOLE[-1])  
value_drill_DRYHOLE = np.linspace(100, -1e6,10)

## Find Min Max for the Vprior Demo plot
vprior_INPUT_min = mymodule.f_VPRIOR([0.9,0.1], value_array, value_drill_DRYHOLE[-1])  
vprior_INPUT_max = mymodule.f_VPRIOR([0.9,0.1], value_array, value_drill_DRYHOLE[0])   
VPI_max = mymodule.Vperfect(Pr_prior_POS_demo, value_array,  value_drill_DRYHOLE[0])   

# Call f_VPRIOR over array value_drill_DRYHOLE
vprior_INPUT_demo_list = list(map(lambda vv: mymodule.f_VPRIOR([1-Pr_prior_POS_demo,Pr_prior_POS_demo], 
                                                              value_array,vv),value_drill_DRYHOLE))
st.subheader('$Pr(Success) = Pr(\Theta=Positive)=$'+str(Pr_prior_POS_demo))  #Pr_prior_POS_demo[0]
st.write('Average outcome, using $Pr(Success)$ ~ Prior probability')
st.write(r'''$V_{prior} =  \max\limits_a \Sigma_{i=1}^2 Pr(\Theta = \theta_i)  v_a(\theta_i) \ \  \forall a $''')

showVperfect = st.checkbox('Show Vperfect')

firstfig, ax = plt.subplots()
plt.plot(value_drill_DRYHOLE, vprior_INPUT_demo_list,'g.-', linewidth=5,label='$V_{prior}$')
plt.ylabel(r'Average Outcome Value [\$]',fontsize=14)
plt.xlabel('Dryhole Cost', color='darkred',fontsize=14)
# axins3 = inset_axes(ax, width="30%", height="30%", loc=2)
#st.write(np.mean(vprior_INPUT_demo_list), np.min(value_drill_DRYHOLE),(VPI_max+20))
txtonplot = r'$v_{a=Drill}(\Theta=Positive) =$'
ax.text(np.min(value_drill_DRYHOLE), value_array[-1,-1]*0.7, txtonplot+'\${:0,.0f}'.format(value_array[-1,-1]), 
        size=12, color='green',
         #va="baseline", ha="left", multialignment="left",
          horizontalalignment='left',
         verticalalignment='top')#, bbox=dict(fc="none"))

if showVperfect:  
    VPIlist = list(map(lambda uu: mymodule.Vperfect(Pr_prior_POS_demo, value_array,uu),value_drill_DRYHOLE))
    # st.write('VPI',np.array(VPIlist),vprior_INPUT_demo_list)
    VOIperfect = np.maximum((np.array(VPIlist)-np.array(vprior_INPUT_demo_list)),np.zeros(len(vprior_INPUT_demo_list)))
    # VPI_list = list(map(lambda v: mymodule.f_Vperfect(Pr_prior_POS_demo, value_array, v), value_drill_DRYHOLE))
    ax.plot(value_drill_DRYHOLE,VPIlist,'b', linewidth=5, alpha=0.5, label='$V_{perfect}$')
    ax.plot(value_drill_DRYHOLE,VOIperfect,'b--', label='$VOI_{perfect}$')

plt.legend(loc=1)
plt.ylim([vprior_INPUT_min,value_array[-1,-1]*0.8]) # YLIM was (VPI_max+20)


# additional code before plt.show()
formatter = ticker.ScalarFormatter()
formatter.set_scientific(False)
# ax.yaxis.set_major_formatter(formatter)
ax.yaxis.set_major_formatter('${x:0,.0f}') #:0,.0f
ax.xaxis.set_major_formatter(formatter)
ax.xaxis.set_major_formatter('${x:0,.0f}')
st.pyplot(firstfig)

if showVperfect:  
    
    st.write('Since you "know" when either subsurface condition occurs, you can pick the best ($\max\limits_a$) drilling altervative first ($v_a$).')
    st.write(r'''$V_{perfect} =  \Sigma_{i=1}^2 Pr(\Theta = \theta_i) \max\limits_a v_a(\theta_i) \ \  \forall a $''')
    st.write(r'''$VOI_{perfect} = V_{perfect}-V_{prior}=$'''+str(VPIlist[0])+' - '+str(vprior_INPUT_demo_list[0]))


with st.sidebar:
            
    # LOCATION OF THIS FILE (Carbonate Aquifer only to start?)
    uploaded_files = st.file_uploader("Choose a Data with Positive Label file (\'POS_\' :fire:) & with Negative (\'NEG_\':thumbsdown:) file",type=['csv'],accept_multiple_files=True)
    st.write(len(uploaded_files))

    if uploaded_files is not None and len(uploaded_files)==2:
        st.subheader('ML Nevada Data')
        st.subheader('Choose attribute for VOI calculation')
        
        for uploaded_file in uploaded_files:
            # bytes_data = uploaded_file.read()
            st.write("filename:", uploaded_file.name)
            if uploaded_file.name[0:3]=='POS':
               pos_upload_file = uploaded_file
               df = pd.read_csv(pos_upload_file)
               attribute0 = st.selectbox('What attributes would you like to calculate', df.columns) 
               st.write('POS File summary...')
               st.write(df.describe())

            elif uploaded_file.name[0:3]=='NEG':
                neg_upload_file = uploaded_file
                dfN = pd.read_csv(neg_upload_file)
                st.write('NEG File preview...')
                st.write(dfN.describe())
            else:
                st.write('Dude, you didn\'t select a POS and NEG file, try again')

        if pos_upload_file.name[3:7] != neg_upload_file.name[3:7]:
                st.write('You aren\'t comparing data from the same region. STOP!')
        # else:        
            
    # attribute0 = st.multiselect('What attributes would you like to calculate', df.columns,max_selections=2)
    
    # with st.echo(): # this prints out 
    # st.write("Update dry hole loss in sidebar.")
    # number = st.number_input(r'''$v_{a=drill}(\theta=Negative)$''')
    # st.write('The current number is ', number)

    #with st.spinner("Loading..."):
    #    time.sleep(5)
    #st.success("Done!")
    else:
        st.write('please upload file')


# uploaded_fileNEG = st.file_uploader("Choose a NEG file",type=['csv'])
st.write('uploaded_files==None attribute0==None', uploaded_files==None)
if uploaded_files is not None:
    
    # df = pd.read_csv(uploaded_file)
    # dfN = pd.read_csv(file_path+neg_upload_file)
    # st.subheader('ML Nevada Data')
    # st.write('File preview...')
    # st.write(df.head())

    # st.write(df)
      
    if attribute0 is not None:
        st.title('You picked this attribute: '+attribute0)
        st.write('Thresholding distances to labels')

        x_cur = attribute0
    
        screen_att0 ='PosSite_Di'
        screen_att1 ='NegSite_Di'
        if any(df.columns.str.contains('GeodeticStrainRate')):
            y_cur0 = 'GeodeticStrainRate'  # hard code for now will come from multiselect
        elif any(df.columns.str.contains('geod_2ndinv_conus117_250m')):
            y_cur0 = 'geod_2ndinv_conus117_250m'
        else:
            st.print('no know strain in df')

        df_screen = df[df[x_cur]>-9999]
        df_screenN = dfN[dfN[x_cur]>-9999]
        st.write('dataframe is shape: {thesize}'.format(thesize=df_screenN.shape))
        #st.write('attribute stats ', df_screen[attribute0].describe())

        neg_site_col_name = 'NegSite_Distance' # I change the name when making csv 'tsite_dist_nvml_neg_conus117_250m' #
        distance_meters = st.slider('!USING THIS! '+neg_site_col_name+' Change likelihood by *screening* distance to positive label [meters]',
                                    10, int(np.max(df_screen['PosSite_Distance'])-10), 800, step=100) # min, max, default
        # NEG_distance_meters = st.slider('Change likelihood by *screening* distance to negative label [km or meters??]', 
        #     10, int(np.max(df_screenN['NegSite_Di'])-10), int(np.median(df_screenN['NegSite_Di'])), step=1000)

        # round to make sure it rounds to nearest 10
        dfpair0 = df_screen[(df_screen['PosSite_Distance'] <=round(distance_meters,-1))] 
        
        dfpair = dfpair0[dfpair0[x_cur]>-9999] 
        # # # OJO : may want to keep this off until have it for NEG 
        dfpairN = df_screenN[(df_screenN[neg_site_col_name ] <=round(distance_meters,-1))] 
        
        if np.shape(dfpairN)[0]==0:
            st.write('using Q1 distance for Negative sites')
            dfpairN = df_screenN[(df_screenN[neg_site_col_name ] <= np.percentile(df_screenN[neg_site_col_name ],10))] 
        
        st.subheader('Calculate & Display Likelihoods')
        st.write('We can compute this "empirical" likelihood with the counts of interpretations.')
       
        #waiting_condition = 1
        #while (waiting_condition):
        #    st.image('https://media.giphy.com/media/gu9XBXiz60HlO5p9Nz/giphy.gif')

        # waiting_condition = mymodule.my_kdeplot(dfpair,x_cur,y_cur0,y_cur1,waiting_condition)
        
        # split up if we want to test bandwidth 
        X_train, X_test, y_train, y_test = mymodule.make_train_test(dfpair,x_cur,dfpairN)
 
        best_params, accuracy = mymodule.optimal_bin(X_train, y_train)

        # Likelihood via KDE estimate
        predictedLikelihood_pos, predictedLikelihood_neg, x_sampled, count_ij= mymodule.likelihood_KDE(X_train,X_test, y_train, y_test,x_cur,y_cur0, best_params)

        #Basic question: How far apart (different) are two distributions P and Q? Measured through distance & divergences
        #https://nobel.web.unc.edu/wp-content/uploads/sites/13591/2020/11/Distance-Divergence.pdf

        
        #st.write('*Given that we know the TRUE GEOTHERMAL OUTCOME (remember "$|$" stands for "given"), what is the likelihood of the label GIVEN the data (X) ')
        #st.subheader(' :violet['+r'''$Pr(\Theta = \theta_i | X =x_j)$'''+'] ~\
        #             :blue['+r'''$Pr(\Theta = \theta_i)$'''+'] \
        #             '+r'''$Pr( X=x_j | \Theta = \theta_i )$''')
          
        st.write(':blue['+r'''$Pr(\Theta = \theta_i)$'''+'] in posterior')
        Pr_prior_POS = mymodule.Prior_probability_binary('Prior used in Posterior')
        st.header('How much is this imperfect data worth?')
        st.subheader(':point_down: :violet[Posterior]~:blue[Prior]:point_up_2: x Likelhood :arrow_heading_up:')
        
        # POSTERIOR via_Naive_Bayes: Draw back here the marginal not using scaled likelihood..
        post_input, post_uniform = mymodule.Posterior_via_NaiveBayes(Pr_prior_POS,X_train, X_test, y_train, y_test, x_sampled, x_cur)

        value_array, value_array_df = mymodule.make_value_array(count_ij, profit_drill_pos= 1e6, cost_drill_neg = -1e6)
        #st.write('value_array', value_array)

        #f_VPRIOR(X_unif_prior, value_array, value_drill_DRYHOLE[-1])  
        value_drill_DRYHOLE = np.linspace(100, -1e6,10)

        # This function can be called with multiple values of "dry hole"
        vprior_unif_out = mymodule.f_VPRIOR([1-Pr_prior_POS,Pr_prior_POS], value_array) #, value_drill_DRYHOLE[-1]       
                       
        #st.subheader(r'''$V_{prior}$ '''+'${:0,.0f}'.format(vprior_unif_out).replace('$-','-$'))

        VPI = mymodule.Vperfect(Pr_prior_POS, value_array)
        # st.subheader(r'''$VOI_{perfect}$ ='''+str(locale.currency(VPI, grouping=True )))
        #st.subheader('Vprior  \${:0,.0f},\t   VOIperfect = \${:0,.0f}'.format(vprior_unif_out,VPI).replace('$-','-$'))
        
        # # DO NOT USEmymodule.marginal( because it's passing unscale likelihood!!!)
        # # Pr_Marg = mymodule.marginal(Pr_prior_POS, predictedLikelihood_pos, predictedLikelihood_neg, x_sampled)
        Pr_InputMarg, Pr_UnifMarg, Prm_d_Input, Prm_d_Uniform = mymodule.Posterior_by_hand(Pr_prior_POS,predictedLikelihood_pos, predictedLikelihood_neg, x_sampled)
        # st.write(np.shape(Pr_Marg),Pr_Marg[0,-20:],Pr_Marg[1,-20:])
        
        mymodule.Posterior_Marginal_plot(Prm_d_Input, Prm_d_Uniform, Pr_InputMarg, x_cur, x_sampled) # WAS inputting: post_input, post_uniform, Pr_Marg, x_cur, x_sampled)

        # VII_unif = mymodule.f_VIMPERFECT(post_uniform, value_array,Pr_UnifMarg)
        
        VII_input = mymodule.f_VIMPERFECT(Prm_d_Input, value_array, Pr_InputMarg)
        VII_unifPrior = mymodule.f_VIMPERFECT(Prm_d_Uniform, value_array, Pr_UnifMarg)
        
        st.latex(r'''\color{purple} Pr( \Theta = \theta_i | X =x_j ) = \color{blue}
            \frac{Pr(\Theta = \theta_i ) \color{black} Pr( X=x_j | \Theta = \theta_i )}{\color{orange} Pr (X=x_j)} 
            ''')
        
        st.subheader(r'''$V_{imperfect}$='''+'${:0,.0f}'.format(VII_input).replace('$-','-$'))
        st.subheader('Vprior  \${:0,.0f},\t   VOIperfect = \${:0,.0f}'.format(vprior_unif_out,VPI).replace('$-','-$'))
        # st.write('with uniform marginal', locale.currency(VII_unifMarginal, grouping=True ))
        st.write('with uniform Prior', '${:0,.0f}'.format(VII_unifPrior).replace('$-','-$'))
        st.write('Using these $v_a(\Theta)$',value_array_df)
        MI_post, NMI_post = mymodule.f_MI(Prm_d_Input,Pr_InputMarg)
        st.write('Mutual Information:', MI_post)
        st.write('Normalized Mutual Information:', NMI_post)
        st.write(accuracy,(VII_input,MI_post,accuracy)) #['bandwidth']
        dataframe4clipboard = pd.DataFrame([[VII_input,NMI_post,accuracy]])#,  columns=['VII','NMI','accuracy'])
        st.write(dataframe4clipboard)
        dataframe4clipboard.to_clipboard(excel=True,index=False)

    else: 
        st.write("Please upload data files on left")
else:
    st.write("Please upload any data.")
        