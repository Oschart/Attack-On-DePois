### Breaking the De-Pois Defense
**Abstract**
   Attacks on machine learning models have been, since their conception, a very persistent and evasive issue resembling an endless cat-and-mouse game. One major variant of such attacks is poisoning attacks which can indirectly manipulate an ML model. It has been observed over the years that the majority of proposed effective defense models are only effective when an attacker is not aware of them being employed. In this paper, we show that the attack-agnostic De-Pois defense is hardly an exception to that rule. In fact, we demonstrate its vulnerability to the simplest White-Box and Black-Box attacks by an attacker that knows the structure of the De-Pois defense model. In essence, the De-Pois defense relies on a critic model that can be used to detect poisoned data before passing it to the target model. In our work, we break this poison-protection layer by replicating the critic model and then performing a composed gradient-sign attack on both the critic and target models simultaneously -- allowing us to bypass the critic firewall to poison the target model.

<kbd>![image](https://drive.google.com/uc?export=view&id=1CsWIP-HO57fUe3THfHXemuqBDzXYVoBI)</kbd>


## Prerequisites:
- Create a conda environemnt using (environment.yml)


## To run the Attack:
```
cd src/
```
```
python full_pipeline.py 
```

