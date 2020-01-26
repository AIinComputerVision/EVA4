# EVA4 S1
Deep Neural Networks code related to Computer vision


1. What are Channels and Kernels?
   
   Channel is a container of similar features(concepts)
   
   Kernels are feature extractors.
 
2. Why we use 3*3 Kernels?
   
   3*3 kernels preserve axis of symmetry and therefore used in all convolution operations.
   
3. How are kernels initialized?
   
   Kernels are initialized with random numbers.
   
4. What happens during the training of DNN?
   
   Pass an input image to the network.
   
   Convolve it with kernels. We get Channels. 
   Downsample the channels.
   
   This process of convolving and downsampling is repeated until the size of the receptive field equals the size of the image.
   So the size of the DNN equals the size of the image.
   Meanwhile the DNN is able to extract edges and textures with the help of kernels. Combining these textures it creates patterns. These patterns make parts of the object. With the linear combination of all these parts of object, an output object is created. The difference between the object obtained by the DNN and the original input image, is the error. This error is backpropagated to the network and the training happens again and again until the error is negligible.
   
5. The process of convolution using python code and consider the image size to be 199*199.

   The convolution happens 99 times until the receptive field is equal to the size of the image.
   
         arr=[i for i in range(200) if i%2 !=0]
   
         arr_rev=arr[::-1]
   
         arr_rev.pop()
   
         print(len(arr_rev))
   
         for i in arr_rev:
         
               print('{} * {} | 3*3 | {} * {}' .format(i,i,i-2,i-2) )
  
  The result is as follows:
  
  99
199 * 199 | 3*3 | 197 * 197

197 * 197 | 3*3 | 195 * 195

195 * 195 | 3*3 | 193 * 193

193 * 193 | 3*3 | 191 * 191

191 * 191 | 3*3 | 189 * 189

189 * 189 | 3*3 | 187 * 187

187 * 187 | 3*3 | 185 * 185

185 * 185 | 3*3 | 183 * 183

183 * 183 | 3*3 | 181 * 181

181 * 181 | 3*3 | 179 * 179

179 * 179 | 3*3 | 177 * 177

177 * 177 | 3*3 | 175 * 175

175 * 175 | 3*3 | 173 * 173

173 * 173 | 3*3 | 171 * 171

171 * 171 | 3*3 | 169 * 169

169 * 169 | 3*3 | 167 * 167

167 * 167 | 3*3 | 165 * 165

165 * 165 | 3*3 | 163 * 163

163 * 163 | 3*3 | 161 * 161

161 * 161 | 3*3 | 159 * 159

159 * 159 | 3*3 | 157 * 157

157 * 157 | 3*3 | 155 * 155

155 * 155 | 3*3 | 153 * 153

153 * 153 | 3*3 | 151 * 151

151 * 151 | 3*3 | 149 * 149

149 * 149 | 3*3 | 147 * 147

147 * 147 | 3*3 | 145 * 145

145 * 145 | 3*3 | 143 * 143

143 * 143 | 3*3 | 141 * 141

141 * 141 | 3*3 | 139 * 139

139 * 139 | 3*3 | 137 * 137

137 * 137 | 3*3 | 135 * 135

135 * 135 | 3*3 | 133 * 133

133 * 133 | 3*3 | 131 * 131

131 * 131 | 3*3 | 129 * 129

129 * 129 | 3*3 | 127 * 127

127 * 127 | 3*3 | 125 * 125

125 * 125 | 3*3 | 123 * 123

123 * 123 | 3*3 | 121 * 121

121 * 121 | 3*3 | 119 * 119

119 * 119 | 3*3 | 117 * 117

117 * 117 | 3*3 | 115 * 115

115 * 115 | 3*3 | 113 * 113

113 * 113 | 3*3 | 111 * 111

111 * 111 | 3*3 | 109 * 109

109 * 109 | 3*3 | 107 * 107

107 * 107 | 3*3 | 105 * 105

105 * 105 | 3*3 | 103 * 103

103 * 103 | 3*3 | 101 * 101

101 * 101 | 3*3 | 99 * 99

99 * 99 | 3*3 | 97 * 97

97 * 97 | 3*3 | 95 * 95

95 * 95 | 3*3 | 93 * 93

93 * 93 | 3*3 | 91 * 91

91 * 91 | 3*3 | 89 * 89

89 * 89 | 3*3 | 87 * 87

87 * 87 | 3*3 | 85 * 85

85 * 85 | 3*3 | 83 * 83

83 * 83 | 3*3 | 81 * 81

81 * 81 | 3*3 | 79 * 79

79 * 79 | 3*3 | 77 * 77

77 * 77 | 3*3 | 75 * 75

75 * 75 | 3*3 | 73 * 73

73 * 73 | 3*3 | 71 * 71

71 * 71 | 3*3 | 69 * 69

69 * 69 | 3*3 | 67 * 67

67 * 67 | 3*3 | 65 * 65

65 * 65 | 3*3 | 63 * 63

63 * 63 | 3*3 | 61 * 61

61 * 61 | 3*3 | 59 * 59

59 * 59 | 3*3 | 57 * 57

57 * 57 | 3*3 | 55 * 55

55 * 55 | 3*3 | 53 * 53

53 * 53 | 3*3 | 51 * 51

51 * 51 | 3*3 | 49 * 49

49 * 49 | 3*3 | 47 * 47

47 * 47 | 3*3 | 45 * 45

45 * 45 | 3*3 | 43 * 43

43 * 43 | 3*3 | 41 * 41

41 * 41 | 3*3 | 39 * 39

39 * 39 | 3*3 | 37 * 37

37 * 37 | 3*3 | 35 * 35

35 * 35 | 3*3 | 33 * 33

33 * 33 | 3*3 | 31 * 31

31 * 31 | 3*3 | 29 * 29

29 * 29 | 3*3 | 27 * 27

27 * 27 | 3*3 | 25 * 25

25 * 25 | 3*3 | 23 * 23

23 * 23 | 3*3 | 21 * 21

21 * 21 | 3*3 | 19 * 19

19 * 19 | 3*3 | 17 * 17

17 * 17 | 3*3 | 15 * 15

15 * 15 | 3*3 | 13 * 13

13 * 13 | 3*3 | 11 * 11

11 * 11 | 3*3 | 9 * 9

9 * 9 | 3*3 | 7 * 7

7 * 7 | 3*3 | 5 * 5

5 * 5 | 3*3 | 3 * 3

3 * 3 | 3*3 | 1 * 1

  
-------------------------------
  
  
 



   
   
 
 
 
